# Security Documentation

## Authentication & Authorization

### User Isolation
All API endpoints are protected by authentication middleware (`get_current_user`) that:
- Validates Firebase ID tokens on every request
- Extracts and verifies user identity (uid) from the token
- Rejects requests with invalid, expired, or revoked tokens
- Returns 401 Unauthorized for authentication failures

### Endpoint Security
All stats and notes endpoints require authentication:
- `GET /api/stats/weekly` - Validates user ID, filters data by authenticated user
- `GET /api/stats/monthly` - Validates user ID, filters data by authenticated user
- `GET /api/stats/lifetime` - Validates user ID, filters data by authenticated user
- `GET /api/notes/today` - Validates user ID, returns only user's notes
- `POST /api/notes/today` - Validates user ID, stores notes with user ID prefix

### Data Isolation
- All database queries filter by `uid` extracted from authenticated user
- Session data is isolated per user using Firestore queries: `.where('uid', '==', uid)`
- Notes are stored with document ID format: `{userId}_{date}` ensuring user isolation
- No user can access another user's data through the API

## Data Protection

### Notes Encryption
- Notes are stored in Firestore with encryption at rest (managed by Google Cloud)
- Firestore automatically encrypts all data at rest using Google-managed encryption keys
- In-transit encryption is enforced via HTTPS/TLS for all API requests

### Aggregation Security
- All statistics aggregations are computed server-side
- Client cannot modify aggregation results
- Aggregations are calculated from source data in real-time
- No cached aggregation data can be manually modified by clients

### Input Validation
- All API endpoints validate input data:
  - Date formats are validated
  - Note text length is limited (max 2000 characters)
  - User ID is extracted from authenticated token (cannot be spoofed)
  - Query parameters are validated and sanitized

## API Request Validation

Every API request:
1. Must include valid `Authorization: Bearer <token>` header
2. Token is validated against Firebase Authentication service
3. User ID is extracted from validated token (not from request body/params)
4. Database queries use authenticated user ID (not client-provided ID)

## Security Best Practices

1. **Never trust client-provided user IDs** - Always extract from authenticated token
2. **Always filter by authenticated user** - Every database query includes `uid` filter
3. **Validate all inputs** - Check data types, lengths, and formats
4. **Use parameterized queries** - Firestore queries are parameterized by design
5. **Log authentication events** - Failed authentication attempts are logged
6. **Handle errors securely** - Don't expose internal details in error messages

## Notes Storage Security

- Notes collection: `daily_notes`
- Document ID format: `{userId}_{date}` (e.g., `user123_2024-01-15`)
- Each document contains:
  - `userId`: User ID (redundant but useful for queries)
  - `date`: Date in YYYY-MM-DD format
  - `notes`: Encrypted text content (Firestore encryption at rest)
  - `createdAt`: Timestamp
  - `updatedAt`: Timestamp

## Session Data Security

- Sessions collection: `listening_sessions`
- All queries filter by `uid` field
- Only sessions with `status: 'STOPPED'` are included in aggregations
- Session data cannot be modified by clients (read-only via stats endpoints)

