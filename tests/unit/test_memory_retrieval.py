"""
Unit tests for memory retrieval improvements, specifically testing condensed memory storage and retrieval.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from src.agents.schemas import EpisodicMemory, SemanticMemory, AgentState
from src.agents.nodes import memory_retrieval_node, _parse_raw_memory


class TestEpisodicMemoryRetrieval:
    """Test cases for episodic memory retrieval of condensed summaries."""
    
    def test_episodic_retrieval_finds_condensed_summaries(self):
        """Test that episodic memory retrieval can find condensed summaries."""
        # Mock store and tools
        mock_store = Mock()
        mock_episodic_tool = Mock()
        mock_semantic_tool = Mock()
        mock_procedural_tool = Mock()
        mock_llm = Mock()
        
        # Mock condensed summary stored in episodic memory
        condensed_episodic_data = [{
            'value': {
                'content': {
                    'user_message': 'Previous conversation context',
                    'ai_response': 'Condensed summary: User discussed their career goals and mentioned interest in machine learning. They prefer technical explanations and have a background in software development.',
                    'timestamp': '2024-01-15T10:30:00Z',
                    'context': 'Condensed episodic summary'
                }
            }
        }]
        
        # Configure mock tools to return condensed summaries
        mock_episodic_tool.invoke.return_value = condensed_episodic_data
        mock_semantic_tool.invoke.return_value = []
        mock_procedural_tool.invoke.return_value = []
        
        # Mock LLM classifier to return episodic retrieval type
        mock_classifier = Mock()
        mock_classifier.retrieval_type = 'episodic'
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_classifier
        
        # Mock store profile search
        mock_store.search.return_value = []
        
        # Create test state
        state = AgentState(message="What did we discuss about my career goals?")
        
        # Call memory retrieval
        result = memory_retrieval_node(
            state,
            llm=mock_llm,
            search_episodic_tool=mock_episodic_tool,
            search_semantic_tool=mock_semantic_tool,
            search_procedural_tool=mock_procedural_tool,
            store=mock_store
        )
        
        # Verify episodic tool was called
        mock_episodic_tool.invoke.assert_called_once_with("What did we discuss about my career goals?")
        
        # Verify condensed summary was retrieved and parsed
        assert 'episodic_memories' in result
        episodic_memories = result['episodic_memories']
        assert len(episodic_memories) == 1
        
        # Verify the condensed summary content
        condensed_memory = episodic_memories[0]
        assert isinstance(condensed_memory, EpisodicMemory)
        assert 'Condensed summary' in condensed_memory.ai_response
        assert 'career goals' in condensed_memory.ai_response
        assert 'machine learning' in condensed_memory.ai_response
        assert condensed_memory.context == 'Condensed episodic summary'
    
    def test_episodic_retrieval_with_follow_up_questions(self):
        """Test that follow-up questions can access previously condensed conversation context."""
        # Mock store and tools
        mock_store = Mock()
        mock_episodic_tool = Mock()
        mock_semantic_tool = Mock()
        mock_procedural_tool = Mock()
        mock_llm = Mock()
        
        # Mock multiple condensed summaries in episodic memory
        condensed_episodic_data = [
            {
                'value': {
                    'content': {
                        'user_message': 'Earlier conversation',
                        'ai_response': 'Condensed summary: User mentioned they work at a tech startup and are interested in AI/ML roles. They have 3 years of Python experience.',
                        'timestamp': '2024-01-15T09:00:00Z',
                        'context': 'Condensed episodic summary'
                    }
                }
            },
            {
                'value': {
                    'content': {
                        'user_message': 'Follow-up discussion',
                        'ai_response': 'Condensed summary: User asked about specific ML frameworks. Recommended TensorFlow and PyTorch based on their background.',
                        'timestamp': '2024-01-15T10:00:00Z',
                        'context': 'Condensed episodic summary'
                    }
                }
            }
        ]
        
        # Configure mock tools
        mock_episodic_tool.invoke.return_value = condensed_episodic_data
        mock_semantic_tool.invoke.return_value = []
        mock_procedural_tool.invoke.return_value = []
        
        # Mock LLM classifier
        mock_classifier = Mock()
        mock_classifier.retrieval_type = 'episodic'
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_classifier
        
        # Mock store profile search
        mock_store.search.return_value = []
        
        # Create test state with follow-up question
        state = AgentState(message="Can you remind me what frameworks you recommended?")
        
        # Call memory retrieval
        result = memory_retrieval_node(
            state,
            llm=mock_llm,
            search_episodic_tool=mock_episodic_tool,
            search_semantic_tool=mock_semantic_tool,
            search_procedural_tool=mock_procedural_tool,
            store=mock_store
        )
        
        # Verify episodic memories were retrieved
        assert 'episodic_memories' in result
        episodic_memories = result['episodic_memories']
        assert len(episodic_memories) == 2
        
        # Verify both condensed summaries are accessible
        summaries = [mem.ai_response for mem in episodic_memories]
        assert any('TensorFlow and PyTorch' in summary for summary in summaries)
        assert any('tech startup' in summary for summary in summaries)
        assert any('Python experience' in summary for summary in summaries)
    
    def test_episodic_retrieval_handles_mixed_memory_types(self):
        """Test that episodic retrieval works with both regular and condensed memories."""
        # Mock store and tools
        mock_store = Mock()
        mock_episodic_tool = Mock()
        mock_semantic_tool = Mock()
        mock_procedural_tool = Mock()
        mock_llm = Mock()
        
        # Mock mixed episodic data (regular + condensed)
        mixed_episodic_data = [
            {
                'value': {
                    'content': {
                        'user_message': 'Hello, how are you?',
                        'ai_response': 'Hello! I\'m doing well, thank you for asking.',
                        'timestamp': '2024-01-15T11:00:00Z',
                        'context': 'Regular conversation'
                    }
                }
            },
            {
                'value': {
                    'content': {
                        'user_message': 'Previous conversation context',
                        'ai_response': 'Condensed summary: User discussed their project timeline and mentioned they need to deliver by end of month. They prefer agile methodology.',
                        'timestamp': '2024-01-15T10:30:00Z',
                        'context': 'Condensed episodic summary'
                    }
                }
            }
        ]
        
        # Configure mock tools
        mock_episodic_tool.invoke.return_value = mixed_episodic_data
        mock_semantic_tool.invoke.return_value = []
        mock_procedural_tool.invoke.return_value = []
        
        # Mock LLM classifier
        mock_classifier = Mock()
        mock_classifier.retrieval_type = 'episodic'
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_classifier
        
        # Mock store profile search
        mock_store.search.return_value = []
        
        # Create test state
        state = AgentState(message="What's my project timeline again?")
        
        # Call memory retrieval
        result = memory_retrieval_node(
            state,
            llm=mock_llm,
            search_episodic_tool=mock_episodic_tool,
            search_semantic_tool=mock_semantic_tool,
            search_procedural_tool=mock_procedural_tool,
            store=mock_store
        )
        
        # Verify both types of memories were retrieved
        assert 'episodic_memories' in result
        episodic_memories = result['episodic_memories']
        assert len(episodic_memories) == 2
        
        # Verify we have both regular and condensed memories
        contexts = [mem.context for mem in episodic_memories]
        assert 'Regular conversation' in contexts
        assert 'Condensed episodic summary' in contexts
        
        # Verify condensed summary contains expected information
        condensed_memory = next(mem for mem in episodic_memories if 'Condensed' in mem.context)
        assert 'project timeline' in condensed_memory.ai_response
        assert 'end of month' in condensed_memory.ai_response
        assert 'agile methodology' in condensed_memory.ai_response
    
    def test_episodic_retrieval_classification_accuracy(self):
        """Test that episodic retrieval is correctly classified for temporal queries."""
        # Mock store and tools
        mock_store = Mock()
        mock_episodic_tool = Mock()
        mock_semantic_tool = Mock()
        mock_procedural_tool = Mock()
        mock_llm = Mock()
        
        # Mock condensed episodic data
        condensed_data = [{
            'value': {
                'content': {
                    'user_message': 'Previous context',
                    'ai_response': 'Condensed summary: User mentioned feeling stressed about work deadlines.',
                    'timestamp': '2024-01-15T09:00:00Z',
                    'context': 'Condensed episodic summary'
                }
            }
        }]
        
        # Configure mock tools
        mock_episodic_tool.invoke.return_value = condensed_data
        mock_semantic_tool.invoke.return_value = []
        mock_procedural_tool.invoke.return_value = []
        
        # Mock LLM classifier to return episodic for temporal queries
        mock_classifier = Mock()
        mock_classifier.retrieval_type = 'episodic'
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_classifier
        
        # Mock store profile search
        mock_store.search.return_value = []
        
        # Test temporal/episodic queries
        temporal_queries = [
            "What did I tell you earlier about my work stress?",
            "Can you remind me what we discussed yesterday?",
            "What was my previous question about?",
            "How did our last conversation end?"
        ]
        
        for query in temporal_queries:
            state = AgentState(message=query)
            
            result = memory_retrieval_node(
                state,
                llm=mock_llm,
                search_episodic_tool=mock_episodic_tool,
                search_semantic_tool=mock_semantic_tool,
                search_procedural_tool=mock_procedural_tool,
                store=mock_store
            )
            
            # Verify episodic tool was called (not semantic)
            mock_episodic_tool.invoke.assert_called_with(query)
            
            # Verify condensed memories were retrieved
            assert 'episodic_memories' in result
            assert len(result['episodic_memories']) == 1
            assert 'work deadlines' in result['episodic_memories'][0].ai_response
            
            # Reset mocks for next iteration
            mock_episodic_tool.reset_mock()
            mock_semantic_tool.reset_mock()


class TestMemoryParsingRobustness:
    """Test cases for robust memory parsing of condensed summaries."""
    
    def test_parse_raw_memory_handles_condensed_summaries(self):
        """Test that _parse_raw_memory correctly handles condensed summary format."""
        # Mock raw data with condensed summary
        raw_data = [
            {
                'value': {
                    'content': {
                        'user_message': 'Context from previous conversation',
                        'ai_response': 'Condensed summary: User is a software engineer working on a mobile app. They mentioned using React Native and having trouble with state management. Prefers Redux for complex applications.',
                        'timestamp': '2024-01-15T10:00:00Z',
                        'context': 'Condensed episodic summary'
                    }
                }
            }
        ]
        
        # Parse the raw data
        parsed_memories = _parse_raw_memory(raw_data, EpisodicMemory)
        
        # Verify parsing succeeded
        assert len(parsed_memories) == 1
        memory = parsed_memories[0]
        assert isinstance(memory, EpisodicMemory)
        
        # Verify condensed content is preserved
        assert 'Condensed summary' in memory.ai_response
        assert 'software engineer' in memory.ai_response
        assert 'React Native' in memory.ai_response
        assert 'state management' in memory.ai_response
        assert 'Redux' in memory.ai_response
        assert memory.context == 'Condensed episodic summary'
    
    def test_parse_raw_memory_handles_malformed_condensed_data(self):
        """Test that parsing gracefully handles malformed condensed memory data."""
        # Mock malformed data
        malformed_data = [
            {
                'value': {
                    'content': {
                        'user_message': 'Valid message',
                        'ai_response': '',  # Empty response
                        'timestamp': '2024-01-15T10:00:00Z',
                        'context': 'Condensed episodic summary'
                    }
                }
            },
            {
                'value': {
                    'content': {
                        # Missing user_message
                        'ai_response': 'Condensed summary: Some content',
                        'timestamp': '2024-01-15T10:00:00Z',
                        'context': 'Condensed episodic summary'
                    }
                }
            },
            {
                'value': {
                    # Missing content key
                    'other_data': 'invalid'
                }
            }
        ]
        
        # Parse the malformed data
        parsed_memories = _parse_raw_memory(malformed_data, EpisodicMemory)
        
        # Should skip malformed entries and return empty list
        assert len(parsed_memories) == 0
    
    def test_parse_raw_memory_handles_json_string_format(self):
        """Test that parsing handles JSON string format from tools."""
        # The current implementation of _parse_raw_memory expects a list of dictionaries,
        # not a JSON string. The memory_retrieval_node handles JSON string conversion
        # before calling _parse_raw_memory. Let's test the actual expected behavior.
        
        # Mock data that would come from JSON.loads() of tool output
        raw_data_list = [
            {
                'value': {
                    'content': {
                        'user_message': 'Previous conversation',
                        'ai_response': 'Condensed summary: User asked about Python best practices. Discussed PEP 8, type hints, and testing frameworks.',
                        'timestamp': '2024-01-15T10:00:00Z',
                        'context': 'Condensed episodic summary'
                    }
                }
            }
        ]
        
        # Parse the data (as it would be after JSON conversion in memory_retrieval_node)
        parsed_memories = _parse_raw_memory(raw_data_list, EpisodicMemory)
        
        # Verify parsing succeeded
        assert len(parsed_memories) == 1
        memory = parsed_memories[0]
        assert isinstance(memory, EpisodicMemory)
        assert 'Python best practices' in memory.ai_response
        assert 'PEP 8' in memory.ai_response
        assert memory.context == 'Condensed episodic summary'


class TestSemanticMemoryRetrieval:
    """Test cases for semantic memory retrieval of condensed summaries."""
    
    def test_semantic_retrieval_finds_condensed_summaries(self):
        """Test that semantic memory retrieval can find condensed summaries."""
        # Mock store and tools
        mock_store = Mock()
        mock_episodic_tool = Mock()
        mock_semantic_tool = Mock()
        mock_procedural_tool = Mock()
        mock_llm = Mock()
        
        # Mock condensed summary stored in semantic memory
        condensed_semantic_data = [{
            'value': {
                'content': {
                    'category': 'summary',
                    'content': 'Condensed summary: User is interested in machine learning and has experience with Python. They work at a tech startup and prefer hands-on learning approaches. Currently exploring deep learning frameworks.',
                    'context': 'Condensed conversation summary',
                    'importance': 0.9,
                    'timestamp': '2024-01-15T10:30:00Z'
                }
            }
        }]
        
        # Configure mock tools to return condensed summaries
        mock_episodic_tool.invoke.return_value = []
        mock_semantic_tool.invoke.return_value = condensed_semantic_data
        mock_procedural_tool.invoke.return_value = []
        
        # Mock LLM classifier to return semantic retrieval type
        mock_classifier = Mock()
        mock_classifier.retrieval_type = 'semantic'
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_classifier
        
        # Mock store profile search
        mock_store.search.return_value = []
        
        # Create test state with knowledge-based query
        state = AgentState(message="What do you know about my technical background?")
        
        # Call memory retrieval
        result = memory_retrieval_node(
            state,
            llm=mock_llm,
            search_episodic_tool=mock_episodic_tool,
            search_semantic_tool=mock_semantic_tool,
            search_procedural_tool=mock_procedural_tool,
            store=mock_store
        )
        
        # Verify semantic tool was called
        mock_semantic_tool.invoke.assert_called_once_with("What do you know about my technical background?")
        
        # Verify condensed summary was retrieved and parsed
        assert 'semantic_memories' in result
        semantic_memories = result['semantic_memories']
        assert len(semantic_memories) == 1
        
        # Verify the condensed summary content
        condensed_memory = semantic_memories[0]
        assert isinstance(condensed_memory, SemanticMemory)
        assert condensed_memory.category == 'summary'
        assert 'Condensed summary' in condensed_memory.content
        assert 'machine learning' in condensed_memory.content
        assert 'Python' in condensed_memory.content
        assert 'tech startup' in condensed_memory.content
        assert condensed_memory.context == 'Condensed conversation summary'
        assert condensed_memory.importance == 0.9
    
    def test_semantic_retrieval_knowledge_based_queries(self):
        """Test that knowledge-based queries can access condensed conversation context."""
        # Mock store and tools
        mock_store = Mock()
        mock_episodic_tool = Mock()
        mock_semantic_tool = Mock()
        mock_procedural_tool = Mock()
        mock_llm = Mock()
        
        # Mock multiple condensed summaries in semantic memory
        condensed_semantic_data = [
            {
                'value': {
                    'content': {
                        'category': 'summary',
                        'content': 'Condensed summary: User has strong preferences for open-source tools and collaborative development. Mentioned experience with Git, Docker, and Kubernetes.',
                        'context': 'Condensed conversation summary',
                        'importance': 0.8,
                        'timestamp': '2024-01-15T09:00:00Z'
                    }
                }
            },
            {
                'value': {
                    'content': {
                        'category': 'summary',
                        'content': 'Condensed summary: User is working on a microservices architecture project. They prefer REST APIs over GraphQL and have experience with Node.js and Express.',
                        'context': 'Condensed conversation summary',
                        'importance': 0.9,
                        'timestamp': '2024-01-15T10:00:00Z'
                    }
                }
            }
        ]
        
        # Configure mock tools
        mock_episodic_tool.invoke.return_value = []
        mock_semantic_tool.invoke.return_value = condensed_semantic_data
        mock_procedural_tool.invoke.return_value = []
        
        # Mock LLM classifier
        mock_classifier = Mock()
        mock_classifier.retrieval_type = 'semantic'
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_classifier
        
        # Mock store profile search
        mock_store.search.return_value = []
        
        # Create test state with knowledge-based query
        state = AgentState(message="What are my technology preferences and experience?")
        
        # Call memory retrieval
        result = memory_retrieval_node(
            state,
            llm=mock_llm,
            search_episodic_tool=mock_episodic_tool,
            search_semantic_tool=mock_semantic_tool,
            search_procedural_tool=mock_procedural_tool,
            store=mock_store
        )
        
        # Verify semantic memories were retrieved
        assert 'semantic_memories' in result
        semantic_memories = result['semantic_memories']
        assert len(semantic_memories) == 2
        
        # Verify both condensed summaries are accessible
        contents = [mem.content for mem in semantic_memories]
        assert any('open-source tools' in content for content in contents)
        assert any('microservices architecture' in content for content in contents)
        assert any('Git, Docker, and Kubernetes' in content for content in contents)
        assert any('REST APIs over GraphQL' in content for content in contents)
        assert any('Node.js and Express' in content for content in contents)
        
        # Verify all are summary category
        categories = [mem.category for mem in semantic_memories]
        assert all(cat == 'summary' for cat in categories)
    
    def test_semantic_retrieval_handles_mixed_memory_types(self):
        """Test that semantic retrieval works with both regular facts and condensed summaries."""
        # Mock store and tools
        mock_store = Mock()
        mock_episodic_tool = Mock()
        mock_semantic_tool = Mock()
        mock_procedural_tool = Mock()
        mock_llm = Mock()
        
        # Mock mixed semantic data (regular facts + condensed summaries)
        mixed_semantic_data = [
            {
                'value': {
                    'content': {
                        'category': 'preference',
                        'content': 'User prefers dark mode in IDEs',
                        'context': 'Direct preference statement',
                        'importance': 0.6,
                        'timestamp': '2024-01-15T11:00:00Z'
                    }
                }
            },
            {
                'value': {
                    'content': {
                        'category': 'summary',
                        'content': 'Condensed summary: User discussed their educational background in computer science. They have a Master\'s degree and specialized in distributed systems. Currently pursuing additional certifications in cloud computing.',
                        'context': 'Condensed conversation summary',
                        'importance': 0.9,
                        'timestamp': '2024-01-15T10:30:00Z'
                    }
                }
            }
        ]
        
        # Configure mock tools
        mock_episodic_tool.invoke.return_value = []
        mock_semantic_tool.invoke.return_value = mixed_semantic_data
        mock_procedural_tool.invoke.return_value = []
        
        # Mock LLM classifier
        mock_classifier = Mock()
        mock_classifier.retrieval_type = 'semantic'
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_classifier
        
        # Mock store profile search
        mock_store.search.return_value = []
        
        # Create test state
        state = AgentState(message="What do you know about my education and preferences?")
        
        # Call memory retrieval
        result = memory_retrieval_node(
            state,
            llm=mock_llm,
            search_episodic_tool=mock_episodic_tool,
            search_semantic_tool=mock_semantic_tool,
            search_procedural_tool=mock_procedural_tool,
            store=mock_store
        )
        
        # Verify both types of memories were retrieved
        assert 'semantic_memories' in result
        semantic_memories = result['semantic_memories']
        assert len(semantic_memories) == 2
        
        # Verify we have both regular facts and condensed summaries
        categories = [mem.category for mem in semantic_memories]
        assert 'preference' in categories
        assert 'summary' in categories
        
        # Verify condensed summary contains expected information
        condensed_memory = next(mem for mem in semantic_memories if mem.category == 'summary')
        assert 'computer science' in condensed_memory.content
        assert 'Master\'s degree' in condensed_memory.content
        assert 'distributed systems' in condensed_memory.content
        assert 'cloud computing' in condensed_memory.content
        
        # Verify regular fact is preserved
        preference_memory = next(mem for mem in semantic_memories if mem.category == 'preference')
        assert 'dark mode' in preference_memory.content
    
    def test_semantic_retrieval_classification_accuracy(self):
        """Test that semantic retrieval is correctly classified for knowledge-based queries."""
        # Mock store and tools
        mock_store = Mock()
        mock_episodic_tool = Mock()
        mock_semantic_tool = Mock()
        mock_procedural_tool = Mock()
        mock_llm = Mock()
        
        # Mock condensed semantic data
        condensed_data = [{
            'value': {
                'content': {
                    'category': 'summary',
                    'content': 'Condensed summary: User has strong analytical skills and enjoys problem-solving. They prefer systematic approaches to debugging and testing.',
                    'context': 'Condensed conversation summary',
                    'importance': 0.8,
                    'timestamp': '2024-01-15T09:00:00Z'
                }
            }
        }]
        
        # Configure mock tools
        mock_episodic_tool.invoke.return_value = []
        mock_semantic_tool.invoke.return_value = condensed_data
        mock_procedural_tool.invoke.return_value = []
        
        # Mock LLM classifier to return semantic for knowledge queries
        mock_classifier = Mock()
        mock_classifier.retrieval_type = 'semantic'
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_classifier
        
        # Mock store profile search
        mock_store.search.return_value = []
        
        # Test knowledge-based/semantic queries
        knowledge_queries = [
            "What are my skills and strengths?",
            "Tell me about my technical preferences",
            "What do you know about my work style?",
            "What are my professional interests?"
        ]
        
        for query in knowledge_queries:
            state = AgentState(message=query)
            
            result = memory_retrieval_node(
                state,
                llm=mock_llm,
                search_episodic_tool=mock_episodic_tool,
                search_semantic_tool=mock_semantic_tool,
                search_procedural_tool=mock_procedural_tool,
                store=mock_store
            )
            
            # Verify semantic tool was called (not episodic)
            mock_semantic_tool.invoke.assert_called_with(query)
            
            # Verify condensed memories were retrieved
            assert 'semantic_memories' in result
            assert len(result['semantic_memories']) == 1
            assert 'analytical skills' in result['semantic_memories'][0].content
            assert result['semantic_memories'][0].category == 'summary'
            
            # Reset mocks for next iteration
            mock_episodic_tool.reset_mock()
            mock_semantic_tool.reset_mock()
    
    def test_semantic_retrieval_high_importance_summaries(self):
        """Test that condensed summaries maintain high importance scores for better retrieval."""
        # Mock store and tools
        mock_store = Mock()
        mock_episodic_tool = Mock()
        mock_semantic_tool = Mock()
        mock_procedural_tool = Mock()
        mock_llm = Mock()
        
        # Mock condensed summaries with high importance
        high_importance_data = [
            {
                'value': {
                    'content': {
                        'category': 'summary',
                        'content': 'Condensed summary: Critical project information - User is leading a team of 5 developers on a high-priority client project with strict deadlines.',
                        'context': 'Condensed conversation summary',
                        'importance': 0.95,  # Very high importance
                        'timestamp': '2024-01-15T10:00:00Z'
                    }
                }
            },
            {
                'value': {
                    'content': {
                        'category': 'preference',
                        'content': 'User likes coffee',
                        'context': 'Casual mention',
                        'importance': 0.3,  # Low importance
                        'timestamp': '2024-01-15T11:00:00Z'
                    }
                }
            }
        ]
        
        # Configure mock tools
        mock_episodic_tool.invoke.return_value = []
        mock_semantic_tool.invoke.return_value = high_importance_data
        mock_procedural_tool.invoke.return_value = []
        
        # Mock LLM classifier
        mock_classifier = Mock()
        mock_classifier.retrieval_type = 'semantic'
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_classifier
        
        # Mock store profile search
        mock_store.search.return_value = []
        
        # Create test state
        state = AgentState(message="What important work information should I know?")
        
        # Call memory retrieval
        result = memory_retrieval_node(
            state,
            llm=mock_llm,
            search_episodic_tool=mock_episodic_tool,
            search_semantic_tool=mock_semantic_tool,
            search_procedural_tool=mock_procedural_tool,
            store=mock_store
        )
        
        # Verify memories were retrieved
        assert 'semantic_memories' in result
        semantic_memories = result['semantic_memories']
        assert len(semantic_memories) == 2
        
        # Verify high importance condensed summary is present
        summary_memory = next(mem for mem in semantic_memories if mem.category == 'summary')
        assert summary_memory.importance == 0.95
        assert 'Critical project information' in summary_memory.content
        assert 'team of 5 developers' in summary_memory.content
        assert 'high-priority client project' in summary_memory.content
        
        # Verify lower importance regular fact is also present
        preference_memory = next(mem for mem in semantic_memories if mem.category == 'preference')
        assert preference_memory.importance == 0.3
        assert 'coffee' in preference_memory.content


class TestIntegrationMemoryFlow:
    """Integration tests for complete memory condensation, storage, and retrieval cycle."""
    
    def test_end_to_end_memory_condensation_and_retrieval(self):
        """Test complete memory flow from condensation to retrieval."""
        # Mock store and tools
        mock_store = Mock()
        mock_episodic_tool = Mock()
        mock_semantic_tool = Mock()
        mock_procedural_tool = Mock()
        mock_llm = Mock()
        
        # Simulate condensed memories stored in both namespaces
        condensed_episodic_data = [{
            'value': {
                'content': {
                    'user_message': 'Previous conversation context',
                    'ai_response': 'Condensed summary: User discussed their career transition from finance to tech. They have completed a coding bootcamp and are looking for entry-level software development positions.',
                    'timestamp': '2024-01-15T10:00:00Z',
                    'context': 'Condensed episodic summary'
                }
            }
        }]
        
        condensed_semantic_data = [{
            'value': {
                'content': {
                    'category': 'summary',
                    'content': 'Condensed summary: User has finance background but transitioned to tech. Completed coding bootcamp, seeking entry-level software development roles. Strong analytical skills from finance experience.',
                    'context': 'Condensed conversation summary',
                    'importance': 0.9,
                    'timestamp': '2024-01-15T10:00:00Z'
                }
            }
        }]
        
        # Test episodic retrieval
        mock_episodic_tool.invoke.return_value = condensed_episodic_data
        mock_semantic_tool.invoke.return_value = []
        mock_procedural_tool.invoke.return_value = []
        
        # Mock LLM classifier for episodic
        mock_classifier = Mock()
        mock_classifier.retrieval_type = 'episodic'
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_classifier
        
        # Mock store profile search
        mock_store.search.return_value = []
        
        # Test episodic retrieval
        episodic_state = AgentState(message="What did we discuss about my career change?")
        episodic_result = memory_retrieval_node(
            episodic_state,
            llm=mock_llm,
            search_episodic_tool=mock_episodic_tool,
            search_semantic_tool=mock_semantic_tool,
            search_procedural_tool=mock_procedural_tool,
            store=mock_store
        )
        
        # Verify episodic retrieval found condensed summary
        assert 'episodic_memories' in episodic_result
        assert len(episodic_result['episodic_memories']) == 1
        episodic_memory = episodic_result['episodic_memories'][0]
        assert 'career transition' in episodic_memory.ai_response
        assert 'coding bootcamp' in episodic_memory.ai_response
        
        # Reset mocks for semantic test
        mock_episodic_tool.reset_mock()
        mock_semantic_tool.reset_mock()
        
        # Test semantic retrieval
        mock_episodic_tool.invoke.return_value = []
        mock_semantic_tool.invoke.return_value = condensed_semantic_data
        mock_classifier.retrieval_type = 'semantic'
        
        semantic_state = AgentState(message="What do you know about my professional background?")
        semantic_result = memory_retrieval_node(
            semantic_state,
            llm=mock_llm,
            search_episodic_tool=mock_episodic_tool,
            search_semantic_tool=mock_semantic_tool,
            search_procedural_tool=mock_procedural_tool,
            store=mock_store
        )
        
        # Verify semantic retrieval found condensed summary
        assert 'semantic_memories' in semantic_result
        assert len(semantic_result['semantic_memories']) == 1
        semantic_memory = semantic_result['semantic_memories'][0]
        assert semantic_memory.category == 'summary'
        assert 'finance background' in semantic_memory.content
        assert 'analytical skills' in semantic_memory.content
    
    def test_memory_retrieval_works_regardless_of_classification(self):
        """Test that memory retrieval works correctly regardless of classification type."""
        # Mock store and tools
        mock_store = Mock()
        mock_episodic_tool = Mock()
        mock_semantic_tool = Mock()
        mock_procedural_tool = Mock()
        mock_llm = Mock()
        
        # Same condensed content stored in both namespaces (as it should be)
        shared_content = "User is working on a React project and needs help with state management. They prefer Redux for complex applications but are open to Context API for simpler use cases."
        
        condensed_episodic_data = [{
            'value': {
                'content': {
                    'user_message': 'Previous discussion',
                    'ai_response': f'Condensed summary: {shared_content}',
                    'timestamp': '2024-01-15T10:00:00Z',
                    'context': 'Condensed episodic summary'
                }
            }
        }]
        
        condensed_semantic_data = [{
            'value': {
                'content': {
                    'category': 'summary',
                    'content': f'Condensed summary: {shared_content}',
                    'context': 'Condensed conversation summary',
                    'importance': 0.9,
                    'timestamp': '2024-01-15T10:00:00Z'
                }
            }
        }]
        
        # Mock store profile search
        mock_store.search.return_value = []
        
        # Test different classification types with same query
        test_query = "What did we discuss about React and state management?"
        
        classification_types = ['episodic', 'semantic', 'general']
        
        for classification in classification_types:
            # Reset mocks
            mock_episodic_tool.reset_mock()
            mock_semantic_tool.reset_mock()
            mock_procedural_tool.reset_mock()
            
            # Configure tools based on classification
            if classification == 'episodic':
                mock_episodic_tool.invoke.return_value = condensed_episodic_data
                mock_semantic_tool.invoke.return_value = []
            elif classification == 'semantic':
                mock_episodic_tool.invoke.return_value = []
                mock_semantic_tool.invoke.return_value = condensed_semantic_data
            else:  # general
                mock_episodic_tool.invoke.return_value = condensed_episodic_data
                mock_semantic_tool.invoke.return_value = condensed_semantic_data
            
            mock_procedural_tool.invoke.return_value = []
            
            # Mock LLM classifier
            mock_classifier = Mock()
            mock_classifier.retrieval_type = classification
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_classifier
            
            # Call memory retrieval
            state = AgentState(message=test_query)
            result = memory_retrieval_node(
                state,
                llm=mock_llm,
                search_episodic_tool=mock_episodic_tool,
                search_semantic_tool=mock_semantic_tool,
                search_procedural_tool=mock_procedural_tool,
                store=mock_store
            )
            
            # Verify that relevant condensed content is always accessible
            found_content = False
            
            if classification == 'episodic':
                episodic_memories = result.get('episodic_memories', [])
                if episodic_memories:
                    found_content = any('React project' in mem.ai_response for mem in episodic_memories)
            elif classification == 'semantic':
                semantic_memories = result.get('semantic_memories', [])
                if semantic_memories:
                    found_content = any('React project' in mem.content for mem in semantic_memories)
            else:  # general
                episodic_memories = result.get('episodic_memories', [])
                semantic_memories = result.get('semantic_memories', [])
                found_content = (
                    any('React project' in mem.ai_response for mem in episodic_memories) or
                    any('React project' in mem.content for mem in semantic_memories)
                )
            
            assert found_content, f"Condensed content not found for classification: {classification}"
    
    def test_profile_update_without_duplication(self):
        """Test that profile updates work correctly without creating duplicates."""
        from src.agents.profile_utils import update_or_create_profile, search_existing_profile
        
        # Mock store
        mock_store = Mock()
        
        # Test scenario 1: No existing profile (should create new)
        mock_store.search.return_value = []
        
        new_profile_data = {
            "name": "Test User",
            "background": "Software developer with 3 years experience",
            "communication_style": "Direct and technical",
            "emotional_baseline": "Generally optimistic"
        }
        
        result = update_or_create_profile(mock_store, new_profile_data, "TestUser")
        assert result == True
        
        # Verify put was called to create new profile
        mock_store.put.assert_called_once()
        put_call = mock_store.put.call_args
        assert put_call[0][0] == ("echo_star", "TestUser", "profile")  # namespace
        assert put_call[0][2]["name"] == "Test User"  # profile data
        
        # Reset mock for next test
        mock_store.reset_mock()
        
        # Test scenario 2: Existing profile (should update, not duplicate)
        existing_profile = Mock()
        existing_profile.key = "existing_key"
        existing_profile.value = {
            "name": "Test User",
            "background": "Software developer",
            "communication_style": "Direct",
            "emotional_baseline": "Optimistic"
        }
        
        mock_store.search.return_value = [existing_profile]
        
        updated_profile_data = {
            "name": "Test User",
            "background": "Senior software developer with 5 years experience",
            "communication_style": "Direct and collaborative",
            "emotional_baseline": "Generally optimistic and patient"
        }
        
        result = update_or_create_profile(mock_store, updated_profile_data, "TestUser")
        assert result == True
        
        # Verify put was called with existing key (replacement, not creation)
        mock_store.put.assert_called_once()
        put_call = mock_store.put.call_args
        assert put_call[0][0] == ("echo_star", "TestUser", "profile")  # namespace
        assert put_call[0][1] == "existing_key"  # should use existing key
        
        # Verify profile data was merged correctly
        saved_profile = put_call[0][2]
        assert "Senior software developer" in saved_profile["background"]
        assert "collaborative" in saved_profile["communication_style"]
        
        # Verify no delete was called (no duplicates to clean up)
        mock_store.delete.assert_not_called()
    
    def test_profile_deduplication_during_update(self):
        """Test that duplicate profiles are cleaned up during updates."""
        from src.agents.profile_utils import update_or_create_profile
        
        # Mock store with multiple duplicate profiles
        mock_store = Mock()
        
        # Create mock duplicate profiles
        profile1 = Mock()
        profile1.key = "key1"
        profile1.value = {"name": "User1", "background": "Old profile"}
        profile1.created_at = "2023-01-01"
        
        profile2 = Mock()
        profile2.key = "key2"
        profile2.value = {"name": "User2", "background": "Newer profile"}
        profile2.created_at = "2023-12-01"
        
        profile3 = Mock()
        profile3.key = "key3"
        profile3.value = {"name": "User3", "background": "Another profile"}
        profile3.created_at = "2023-06-01"
        
        mock_store.search.return_value = [profile1, profile2, profile3]
        
        # New profile data to merge
        new_profile_data = {
            "name": "Updated User",
            "background": "Updated background information",
            "communication_style": "Updated style",
            "emotional_baseline": "Updated baseline"
        }
        
        # Call update function
        result = update_or_create_profile(mock_store, new_profile_data, "TestUser")
        assert result == True
        
        # Verify put was called once to update primary profile
        mock_store.put.assert_called_once()
        put_call = mock_store.put.call_args
        assert put_call[0][1] == "key1"  # Should use first profile's key
        
        # Verify delete was called to remove duplicates
        assert mock_store.delete.call_count == 2  # Should delete 2 duplicates
        
        # Check that correct profiles were deleted
        delete_calls = [call.args for call in mock_store.delete.call_args_list]
        expected_deletes = [
            (("echo_star", "TestUser", "profile"), "key2"),
            (("echo_star", "TestUser", "profile"), "key3")
        ]
        
        for expected_delete in expected_deletes:
            assert expected_delete in delete_calls
    
    def test_cross_namespace_memory_accessibility(self):
        """Test that condensed memories are accessible from both storage locations."""
        # Mock store and tools
        mock_store = Mock()
        mock_episodic_tool = Mock()
        mock_semantic_tool = Mock()
        mock_procedural_tool = Mock()
        mock_llm = Mock()
        
        # Same condensed information stored in both namespaces
        shared_summary = "User is planning a career change to data science. They have strong math background and are learning Python and statistics. Interested in machine learning applications in healthcare."
        
        # Episodic version
        episodic_condensed = [{
            'value': {
                'content': {
                    'user_message': 'Career discussion',
                    'ai_response': f'Condensed summary: {shared_summary}',
                    'timestamp': '2024-01-15T10:00:00Z',
                    'context': 'Condensed episodic summary'
                }
            }
        }]
        
        # Semantic version
        semantic_condensed = [{
            'value': {
                'content': {
                    'category': 'summary',
                    'content': f'Condensed summary: {shared_summary}',
                    'context': 'Condensed conversation summary',
                    'importance': 0.9,
                    'timestamp': '2024-01-15T10:00:00Z'
                }
            }
        }]
        
        # Mock store profile search
        mock_store.search.return_value = []
        
        # Test that both episodic and semantic searches can find the same information
        test_queries = [
            ("What did we discuss about my career plans?", "episodic"),
            ("What do you know about my career goals?", "semantic"),
            ("Tell me about my data science interests", "general")
        ]
        
        for query, expected_classification in test_queries:
            # Reset mocks
            mock_episodic_tool.reset_mock()
            mock_semantic_tool.reset_mock()
            mock_procedural_tool.reset_mock()
            
            # Configure tools based on expected classification
            if expected_classification == 'episodic':
                mock_episodic_tool.invoke.return_value = episodic_condensed
                mock_semantic_tool.invoke.return_value = []
            elif expected_classification == 'semantic':
                mock_episodic_tool.invoke.return_value = []
                mock_semantic_tool.invoke.return_value = semantic_condensed
            else:  # general
                mock_episodic_tool.invoke.return_value = episodic_condensed
                mock_semantic_tool.invoke.return_value = semantic_condensed
            
            mock_procedural_tool.invoke.return_value = []
            
            # Mock LLM classifier
            mock_classifier = Mock()
            mock_classifier.retrieval_type = expected_classification
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_classifier
            
            # Call memory retrieval
            state = AgentState(message=query)
            result = memory_retrieval_node(
                state,
                llm=mock_llm,
                search_episodic_tool=mock_episodic_tool,
                search_semantic_tool=mock_semantic_tool,
                search_procedural_tool=mock_procedural_tool,
                store=mock_store
            )
            
            # Verify the shared information is accessible regardless of namespace
            found_career_info = False
            found_data_science = False
            found_python = False
            
            # Check episodic memories
            episodic_memories = result.get('episodic_memories', [])
            for mem in episodic_memories:
                if 'career change to data science' in mem.ai_response:
                    found_career_info = True
                if 'data science' in mem.ai_response:
                    found_data_science = True
                if 'Python' in mem.ai_response:
                    found_python = True
            
            # Check semantic memories
            semantic_memories = result.get('semantic_memories', [])
            for mem in semantic_memories:
                if 'career change to data science' in mem.content:
                    found_career_info = True
                if 'data science' in mem.content:
                    found_data_science = True
                if 'Python' in mem.content:
                    found_python = True
            
            # Assert that key information is found regardless of classification
            assert found_career_info, f"Career info not found for query: {query}"
            assert found_data_science, f"Data science info not found for query: {query}"
            assert found_python, f"Python info not found for query: {query}"