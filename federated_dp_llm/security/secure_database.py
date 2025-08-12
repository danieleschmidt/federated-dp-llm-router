#!/usr/bin/env python3
"""
Secure Database Manager
Addresses SQL injection vulnerabilities with parameterized queries
and secure database operations
"""

import sqlite3
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from contextlib import asynccontextmanager
import json
from dataclasses import dataclass, asdict
from federated_dp_llm.security.secure_config_manager import get_secret

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Secure query result wrapper"""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    affected_rows: int = 0
    error: Optional[str] = None

class SecureDatabase:
    """
    Secure database manager that prevents SQL injection vulnerabilities
    through parameterized queries and input validation
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or ":memory:"
        self.connection_pool = []
        self.max_connections = 10
        self.connection_lock = asyncio.Lock()
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize secure database with proper schema"""
        if self.initialized:
            return
            
        async with self._get_connection() as conn:
            # Create tables with proper constraints
            await self._execute_ddl(conn, """
                CREATE TABLE IF NOT EXISTS user_budgets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL UNIQUE,
                    total_budget REAL NOT NULL DEFAULT 0.0,
                    spent_budget REAL NOT NULL DEFAULT 0.0,
                    remaining_budget REAL NOT NULL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    department TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT valid_budgets CHECK (
                        total_budget >= 0 AND 
                        spent_budget >= 0 AND 
                        remaining_budget >= 0 AND
                        spent_budget <= total_budget
                    )
                )
            """)
            
            await self._execute_ddl(conn, """
                CREATE TABLE IF NOT EXISTS privacy_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    epsilon_spent REAL NOT NULL,
                    query_hash TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    node_id TEXT,
                    department TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_budgets(user_id),
                    CONSTRAINT valid_epsilon CHECK (epsilon_spent > 0)
                )
            """)
            
            await self._execute_ddl(conn, """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    resource TEXT,
                    action TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    success BOOLEAN NOT NULL DEFAULT TRUE,
                    error_message TEXT,
                    additional_data TEXT
                )
            """)
            
            # Create indexes for performance
            await self._execute_ddl(conn, "CREATE INDEX IF NOT EXISTS idx_user_budgets_user_id ON user_budgets(user_id)")
            await self._execute_ddl(conn, "CREATE INDEX IF NOT EXISTS idx_privacy_transactions_user_id ON privacy_transactions(user_id)")
            await self._execute_ddl(conn, "CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp)")
            
        self.initialized = True
        logger.info("Secure database initialized with proper schema and constraints")
    
    @asynccontextmanager
    async def _get_connection(self):
        """Get database connection from pool"""
        async with self.connection_lock:
            if self.connection_pool:
                conn = self.connection_pool.pop()
            else:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.row_factory = sqlite3.Row  # For dict-like access
                
        try:
            yield conn
        finally:
            async with self.connection_lock:
                if len(self.connection_pool) < self.max_connections:
                    self.connection_pool.append(conn)
                else:
                    conn.close()
    
    async def _execute_ddl(self, conn: sqlite3.Connection, sql: str) -> None:
        """Execute DDL statements safely"""
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"DDL execution failed: {e}")
            raise
    
    async def get_user_budget(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user budget using parameterized query to prevent SQL injection
        """
        if not await self._validate_user_id(user_id):
            logger.warning(f"Invalid user_id format: {user_id}")
            return None
            
        async with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Using parameterized query to prevent SQL injection
            cursor.execute("""
                SELECT user_id, total_budget, spent_budget, remaining_budget, 
                       last_updated, department, created_at
                FROM user_budgets 
                WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    async def update_user_budget(self, user_id: str, epsilon_spent: float, 
                               department: Optional[str] = None) -> QueryResult:
        """
        Update user budget with parameterized queries
        """
        if not await self._validate_user_id(user_id):
            return QueryResult(success=False, error="Invalid user ID format")
            
        if epsilon_spent < 0:
            return QueryResult(success=False, error="Epsilon spent cannot be negative")
        
        async with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # First, get current budget
                cursor.execute("""
                    SELECT total_budget, spent_budget 
                    FROM user_budgets 
                    WHERE user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                if not row:
                    return QueryResult(success=False, error="User budget not found")
                
                current_spent = row['spent_budget']
                total_budget = row['total_budget']
                new_spent = current_spent + epsilon_spent
                new_remaining = total_budget - new_spent
                
                if new_remaining < 0:
                    return QueryResult(success=False, error="Insufficient privacy budget")
                
                # Update budget with parameterized query
                cursor.execute("""
                    UPDATE user_budgets 
                    SET spent_budget = ?, 
                        remaining_budget = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (new_spent, new_remaining, user_id))
                
                # Record transaction
                cursor.execute("""
                    INSERT INTO privacy_transactions 
                    (user_id, epsilon_spent, department, timestamp)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (user_id, epsilon_spent, department))
                
                conn.commit()
                
                return QueryResult(
                    success=True,
                    affected_rows=cursor.rowcount,
                    data=[{
                        'user_id': user_id,
                        'spent_budget': new_spent,
                        'remaining_budget': new_remaining
                    }]
                )
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to update user budget: {e}")
                return QueryResult(success=False, error=str(e))
    
    async def create_user_budget(self, user_id: str, total_budget: float,
                               department: Optional[str] = None) -> QueryResult:
        """
        Create new user budget with parameterized queries
        """
        if not await self._validate_user_id(user_id):
            return QueryResult(success=False, error="Invalid user ID format")
            
        if total_budget <= 0:
            return QueryResult(success=False, error="Total budget must be positive")
        
        async with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO user_budgets 
                    (user_id, total_budget, spent_budget, remaining_budget, department)
                    VALUES (?, ?, 0.0, ?, ?)
                """, (user_id, total_budget, total_budget, department))
                
                conn.commit()
                
                return QueryResult(
                    success=True,
                    affected_rows=cursor.rowcount,
                    data=[{
                        'user_id': user_id,
                        'total_budget': total_budget,
                        'spent_budget': 0.0,
                        'remaining_budget': total_budget,
                        'department': department
                    }]
                )
                
            except sqlite3.IntegrityError as e:
                conn.rollback()
                return QueryResult(success=False, error=f"User already exists: {e}")
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to create user budget: {e}")
                return QueryResult(success=False, error=str(e))
    
    async def get_user_transactions(self, user_id: str, limit: int = 100) -> QueryResult:
        """
        Get user privacy transactions with parameterized queries
        """
        if not await self._validate_user_id(user_id):
            return QueryResult(success=False, error="Invalid user ID format")
        
        if limit <= 0 or limit > 1000:  # Prevent excessive data retrieval
            limit = 100
        
        async with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, user_id, epsilon_spent, query_hash, timestamp, 
                           node_id, department
                    FROM privacy_transactions
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (user_id, limit))
                
                rows = cursor.fetchall()
                data = [dict(row) for row in rows]
                
                return QueryResult(success=True, data=data)
                
            except Exception as e:
                logger.error(f"Failed to get user transactions: {e}")
                return QueryResult(success=False, error=str(e))
    
    async def log_audit_event(self, event_type: str, action: str, 
                            user_id: Optional[str] = None,
                            resource: Optional[str] = None,
                            success: bool = True,
                            error_message: Optional[str] = None,
                            additional_data: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        Log audit events with parameterized queries
        """
        async with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                
                additional_data_json = json.dumps(additional_data) if additional_data else None
                
                cursor.execute("""
                    INSERT INTO audit_log 
                    (event_type, user_id, resource, action, success, 
                     error_message, additional_data, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (event_type, user_id, resource, action, success, 
                      error_message, additional_data_json))
                
                conn.commit()
                
                return QueryResult(success=True, affected_rows=cursor.rowcount)
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to log audit event: {e}")
                return QueryResult(success=False, error=str(e))
    
    async def get_audit_logs(self, limit: int = 100, 
                           event_type: Optional[str] = None,
                           user_id: Optional[str] = None) -> QueryResult:
        """
        Get audit logs with parameterized queries and filtering
        """
        if limit <= 0 or limit > 1000:
            limit = 100
        
        async with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # Build query with proper parameterization
                base_query = """
                    SELECT id, event_type, user_id, resource, action, timestamp,
                           success, error_message, additional_data
                    FROM audit_log
                """
                
                params = []
                conditions = []
                
                if event_type:
                    conditions.append("event_type = ?")
                    params.append(event_type)
                    
                if user_id:
                    if not await self._validate_user_id(user_id):
                        return QueryResult(success=False, error="Invalid user ID format")
                    conditions.append("user_id = ?")
                    params.append(user_id)
                
                if conditions:
                    base_query += " WHERE " + " AND ".join(conditions)
                
                base_query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(base_query, params)
                
                rows = cursor.fetchall()
                data = []
                
                for row in rows:
                    row_dict = dict(row)
                    # Parse additional_data JSON if present
                    if row_dict['additional_data']:
                        try:
                            row_dict['additional_data'] = json.loads(row_dict['additional_data'])
                        except json.JSONDecodeError:
                            pass  # Keep as string if JSON parsing fails
                    data.append(row_dict)
                
                return QueryResult(success=True, data=data)
                
            except Exception as e:
                logger.error(f"Failed to get audit logs: {e}")
                return QueryResult(success=False, error=str(e))
    
    async def _validate_user_id(self, user_id: str) -> bool:
        """
        Validate user ID format to prevent injection attacks
        """
        if not user_id or not isinstance(user_id, str):
            return False
            
        # Only allow alphanumeric characters, underscores, and hyphens
        # Length between 3 and 50 characters
        if len(user_id) < 3 or len(user_id) > 50:
            return False
            
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-@.")
        if not all(c in allowed_chars for c in user_id):
            return False
        
        # Prevent SQL injection patterns
        dangerous_patterns = [
            "'", '"', ';', '--', '/*', '*/', 'DROP', 'DELETE', 'UPDATE', 
            'INSERT', 'SELECT', 'UNION', 'OR 1=1', 'OR 1 = 1'
        ]
        
        user_id_upper = user_id.upper()
        for pattern in dangerous_patterns:
            if pattern in user_id_upper:
                return False
        
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check
        """
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Test basic connectivity
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                
                if not result or result[0] != 1:
                    return {
                        'status': 'unhealthy',
                        'error': 'Basic connectivity test failed'
                    }
                
                # Check table integrity
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('user_budgets', 'privacy_transactions', 'audit_log')
                """)
                
                tables = [row[0] for row in cursor.fetchall()]
                expected_tables = {'user_budgets', 'privacy_transactions', 'audit_log'}
                
                if not expected_tables.issubset(set(tables)):
                    return {
                        'status': 'unhealthy',
                        'error': f'Missing tables: {expected_tables - set(tables)}'
                    }
                
                return {
                    'status': 'healthy',
                    'tables': tables,
                    'connection_pool_size': len(self.connection_pool)
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

# Global secure database instance
_secure_db = None

async def get_secure_database() -> SecureDatabase:
    """Get global secure database instance"""
    global _secure_db
    if _secure_db is None:
        _secure_db = SecureDatabase()
        await _secure_db.initialize()
    return _secure_db