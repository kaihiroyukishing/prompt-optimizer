# API Documentation

This document describes the REST API endpoints for the Prompt Optimizer backend.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## Authentication

Currently, the API does not require authentication. Future versions may implement API key authentication.

## Response Format

All API responses follow this format:

```json
{
  "success": true,
  "data": { ... },
  "message": "Optional message",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

Error responses:

```json
{
  "success": false,
  "error": "Error message",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Endpoints

### Optimization

#### Optimize Prompt
```http
POST /api/v1/optimize
```

**Request Body:**
```json
{
  "prompt": "Your original prompt here",
  "session_id": "unique-session-id"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "original_prompt": "Your original prompt here",
    "optimized_prompt": "Optimized version of your prompt",
    "session_id": "unique-session-id",
    "similar_prompts": [
      {
        "prompt": "Similar prompt 1",
        "similarity_score": 0.85
      }
    ],
    "context_used": true
  }
}
```

#### Get Session Prompts
```http
GET /api/v1/prompts/{session_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "unique-session-id",
    "prompts": [
      {
        "id": 1,
        "original_prompt": "Original prompt",
        "optimized_prompt": "Optimized prompt",
        "created_at": "2024-01-01T00:00:00Z"
      }
    ]
  }
}
```

### Context Management

#### Save Context
```http
POST /api/v1/save-context
```

**Request Body:**
```json
{
  "prompt": "Original prompt",
  "optimized_prompt": "Optimized prompt",
  "session_id": "unique-session-id",
  "embedding": [0.1, 0.2, 0.3, ...]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Context saved successfully",
    "session_id": "unique-session-id",
    "prompt_id": "generated-id"
  }
}
```

#### Get Context
```http
GET /api/v1/context/{session_id}?limit=10
```

**Query Parameters:**
- `limit` (optional): Maximum number of contexts to return (default: 10)

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "unique-session-id",
    "contexts": [
      {
        "id": 1,
        "prompt": "Original prompt",
        "optimized_prompt": "Optimized prompt",
        "created_at": "2024-01-01T00:00:00Z"
      }
    ],
    "total_count": 1
  }
}
```

#### Clear Context
```http
DELETE /api/v1/context/{session_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Context cleared for session unique-session-id",
    "session_id": "unique-session-id"
  }
}
```

### System

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "prompt-optimizer-api",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### API Information
```http
GET /
```

**Response:**
```json
{
  "message": "Prompt Optimizer API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `DATABASE_ERROR` | Database operation failed |
| `API_ERROR` | External API call failed |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded |
| `INTERNAL_ERROR` | Internal server error |

## Rate Limiting

- **Default**: 60 requests per minute per IP
- **Headers**: Rate limit information is included in response headers
- **Exceeded**: Returns 429 status code with rate limit error

## Interactive Documentation

When running the development server, interactive API documentation is available at:
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
