# Database Directory

This directory is prepared for persistent database integration to replace the current InMemoryStore.

## Purpose

Currently, EchoStar uses LangGraph's `InMemoryStore` for memory management, which means all memories are lost when the application restarts. This directory is ready for implementing persistent storage solutions.

## Planned Integration

### Database Options

- **PostgreSQL** - Robust relational database with JSON support
- **MongoDB** - Document database for flexible schema
- **SQLite** - Lightweight option for development/small deployments
- **Redis** - In-memory database with persistence options

### Implementation Plan

1. **Database Schema Design** - Define tables/collections for different memory types
2. **Connection Management** - Database connection pooling and configuration
3. **Migration Scripts** - Database setup and schema migrations
4. **Memory Adapters** - Replace InMemoryStore with database-backed storage
5. **Performance Optimization** - Indexing and query optimization

### Files to be Added

```
database/
├── migrations/           # Database migration scripts
├── models/              # Database models/schemas
├── adapters/            # Database adapter implementations
├── config/              # Database-specific configuration
└── scripts/             # Database utility scripts
```

## Memory Types to Persist

1. **Episodic Memories** - Conversation history and interactions
2. **Semantic Memories** - User preferences, facts, and traits
3. **Procedural Memories** - Learned behaviors and rules
4. **User Profiles** - User information and preferences

## Benefits of Persistent Storage

- **Continuity** - Memories persist across application restarts
- **Scalability** - Handle large amounts of memory data
- **Backup & Recovery** - Data protection and disaster recovery
- **Analytics** - Query and analyze conversation patterns
- **Multi-user Support** - Separate memory spaces for different users
