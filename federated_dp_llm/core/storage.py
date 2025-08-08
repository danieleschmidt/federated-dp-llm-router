"""
Simple Storage System for Privacy Budgets

Provides persistent storage for privacy budgets, user data, and system state.
"""

import json
import sqlite3
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading


@dataclass
class UserBudgetRecord:
    """Record for user privacy budget."""
    user_id: str
    department: str
    total_budget: float
    spent_budget: float
    remaining_budget: float
    last_reset: datetime
    created_at: datetime
    updated_at: datetime


@dataclass
class BudgetTransaction:
    """Record for budget spending transaction."""
    transaction_id: str
    user_id: str
    amount: float
    timestamp: datetime
    request_id: Optional[str] = None
    description: Optional[str] = None


class SimpleBudgetStorage:
    """Simple file-based storage for privacy budgets."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.budget_file = self.data_dir / "privacy_budgets.json"
        self.transactions_file = self.data_dir / "transactions.json"
        self.logger = logging.getLogger(__name__)
        
        # In-memory cache
        self._budget_cache: Dict[str, UserBudgetRecord] = {}
        self._transactions: List[BudgetTransaction] = []
        self._lock = threading.RLock()
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load data from files."""
        try:
            # Load budgets
            if self.budget_file.exists():
                with open(self.budget_file, 'r') as f:
                    budget_data = json.load(f)
                    for user_id, data in budget_data.items():
                        # Convert datetime strings back to datetime objects
                        data['last_reset'] = datetime.fromisoformat(data['last_reset'])
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                        self._budget_cache[user_id] = UserBudgetRecord(**data)
            
            # Load transactions
            if self.transactions_file.exists():
                with open(self.transactions_file, 'r') as f:
                    transaction_data = json.load(f)
                    for data in transaction_data:
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                        self._transactions.append(BudgetTransaction(**data))
                        
            self.logger.info(f"Loaded {len(self._budget_cache)} budget records and {len(self._transactions)} transactions")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save data to files."""
        try:
            with self._lock:
                # Save budgets
                budget_data = {}
                for user_id, record in self._budget_cache.items():
                    data = asdict(record)
                    # Convert datetime objects to strings
                    data['last_reset'] = data['last_reset'].isoformat()
                    data['created_at'] = data['created_at'].isoformat()
                    data['updated_at'] = data['updated_at'].isoformat()
                    budget_data[user_id] = data
                
                with open(self.budget_file, 'w') as f:
                    json.dump(budget_data, f, indent=2)
                
                # Save transactions
                transaction_data = []
                for transaction in self._transactions:
                    data = asdict(transaction)
                    data['timestamp'] = data['timestamp'].isoformat()
                    transaction_data.append(data)
                
                with open(self.transactions_file, 'w') as f:
                    json.dump(transaction_data, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
    
    def create_user_budget(self, user_id: str, department: str, total_budget: float = 10.0) -> UserBudgetRecord:
        """Create a new user budget record."""
        with self._lock:
            if user_id in self._budget_cache:
                return self._budget_cache[user_id]
            
            now = datetime.now()
            record = UserBudgetRecord(
                user_id=user_id,
                department=department,
                total_budget=total_budget,
                spent_budget=0.0,
                remaining_budget=total_budget,
                last_reset=now,
                created_at=now,
                updated_at=now
            )
            
            self._budget_cache[user_id] = record
            self._save_data()
            
            self.logger.info(f"Created budget for user {user_id}: {total_budget}")
            return record
    
    def get_user_budget(self, user_id: str) -> Optional[UserBudgetRecord]:
        """Get user budget record."""
        with self._lock:
            return self._budget_cache.get(user_id)
    
    def update_user_budget(self, user_id: str, spent_amount: float, request_id: Optional[str] = None) -> bool:
        """Update user budget by spending amount."""
        with self._lock:
            if user_id not in self._budget_cache:
                self.logger.warning(f"User {user_id} not found, creating default budget")
                self.create_user_budget(user_id, "unknown", 10.0)
            
            record = self._budget_cache[user_id]
            
            if record.remaining_budget < spent_amount:
                self.logger.warning(f"Insufficient budget for user {user_id}: {record.remaining_budget} < {spent_amount}")
                return False
            
            # Update budget
            record.spent_budget += spent_amount
            record.remaining_budget = record.total_budget - record.spent_budget
            record.updated_at = datetime.now()
            
            # Create transaction record
            transaction = BudgetTransaction(
                transaction_id=f"tx_{int(datetime.now().timestamp())}_{user_id}",
                user_id=user_id,
                amount=spent_amount,
                timestamp=datetime.now(),
                request_id=request_id,
                description=f"Privacy budget spent: {spent_amount}"
            )
            
            self._transactions.append(transaction)
            
            # Save data
            self._save_data()
            
            self.logger.info(f"Updated budget for user {user_id}: spent {spent_amount}, remaining {record.remaining_budget}")
            return True
    
    def reset_user_budget(self, user_id: str) -> bool:
        """Reset user's privacy budget."""
        with self._lock:
            if user_id not in self._budget_cache:
                return False
            
            record = self._budget_cache[user_id]
            record.spent_budget = 0.0
            record.remaining_budget = record.total_budget
            record.last_reset = datetime.now()
            record.updated_at = datetime.now()
            
            self._save_data()
            
            self.logger.info(f"Reset budget for user {user_id}")
            return True
    
    def get_user_transactions(self, user_id: str, limit: int = 100) -> List[BudgetTransaction]:
        """Get transaction history for user."""
        with self._lock:
            user_transactions = [tx for tx in self._transactions if tx.user_id == user_id]
            return sorted(user_transactions, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_all_budgets(self) -> Dict[str, UserBudgetRecord]:
        """Get all user budget records."""
        with self._lock:
            return self._budget_cache.copy()
    
    def cleanup_old_transactions(self, days: int = 30):
        """Remove transactions older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._lock:
            original_count = len(self._transactions)
            self._transactions = [tx for tx in self._transactions if tx.timestamp > cutoff]
            removed_count = original_count - len(self._transactions)
            
            if removed_count > 0:
                self._save_data()
                self.logger.info(f"Cleaned up {removed_count} old transactions")


class SQLiteBudgetStorage:
    """SQLite-based storage for privacy budgets (more robust option)."""
    
    def __init__(self, db_path: str = "./data/privacy_budgets.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create budgets table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_budgets (
                        user_id TEXT PRIMARY KEY,
                        department TEXT NOT NULL,
                        total_budget REAL NOT NULL,
                        spent_budget REAL NOT NULL DEFAULT 0.0,
                        remaining_budget REAL NOT NULL,
                        last_reset TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                ''')
                
                # Create transactions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS budget_transactions (
                        transaction_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        amount REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        request_id TEXT,
                        description TEXT,
                        FOREIGN KEY (user_id) REFERENCES user_budgets (user_id)
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON budget_transactions (user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON budget_transactions (timestamp)')
                
                conn.commit()
                self.logger.info("SQLite database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def create_user_budget(self, user_id: str, department: str, total_budget: float = 10.0) -> UserBudgetRecord:
        """Create a new user budget record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                now = datetime.now().isoformat()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO user_budgets 
                    (user_id, department, total_budget, spent_budget, remaining_budget, 
                     last_reset, created_at, updated_at)
                    VALUES (?, ?, ?, 0.0, ?, ?, ?, ?)
                ''', (user_id, department, total_budget, total_budget, now, now, now))
                
                conn.commit()
                
                return UserBudgetRecord(
                    user_id=user_id,
                    department=department,
                    total_budget=total_budget,
                    spent_budget=0.0,
                    remaining_budget=total_budget,
                    last_reset=datetime.fromisoformat(now),
                    created_at=datetime.fromisoformat(now),
                    updated_at=datetime.fromisoformat(now)
                )
                
        except Exception as e:
            self.logger.error(f"Error creating user budget: {e}")
            raise
    
    def get_user_budget(self, user_id: str) -> Optional[UserBudgetRecord]:
        """Get user budget record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM user_budgets WHERE user_id = ?', (user_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return UserBudgetRecord(
                    user_id=row[0],
                    department=row[1],
                    total_budget=row[2],
                    spent_budget=row[3],
                    remaining_budget=row[4],
                    last_reset=datetime.fromisoformat(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7])
                )
                
        except Exception as e:
            self.logger.error(f"Error getting user budget: {e}")
            return None
    
    def update_user_budget(self, user_id: str, spent_amount: float, request_id: Optional[str] = None) -> bool:
        """Update user budget by spending amount."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check current budget
                cursor.execute('SELECT remaining_budget FROM user_budgets WHERE user_id = ?', (user_id,))
                row = cursor.fetchone()
                
                if not row or row[0] < spent_amount:
                    return False
                
                # Update budget
                now = datetime.now().isoformat()
                cursor.execute('''
                    UPDATE user_budgets 
                    SET spent_budget = spent_budget + ?, 
                        remaining_budget = remaining_budget - ?,
                        updated_at = ?
                    WHERE user_id = ?
                ''', (spent_amount, spent_amount, now, user_id))
                
                # Insert transaction
                transaction_id = f"tx_{int(datetime.now().timestamp())}_{user_id}"
                cursor.execute('''
                    INSERT INTO budget_transactions 
                    (transaction_id, user_id, amount, timestamp, request_id, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (transaction_id, user_id, spent_amount, now, request_id, f"Privacy budget spent: {spent_amount}"))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating user budget: {e}")
            return False
    
    def reset_user_budget(self, user_id: str) -> bool:
        """Reset user's privacy budget."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                now = datetime.now().isoformat()
                cursor.execute('''
                    UPDATE user_budgets 
                    SET spent_budget = 0.0,
                        remaining_budget = total_budget,
                        last_reset = ?,
                        updated_at = ?
                    WHERE user_id = ?
                ''', (now, now, user_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"Error resetting user budget: {e}")
            return False


# Global storage instance
_storage_instance: Optional[SimpleBudgetStorage] = None


def get_budget_storage(storage_type: str = "simple", **kwargs) -> SimpleBudgetStorage:
    """Get or create the global budget storage instance."""
    global _storage_instance
    
    if _storage_instance is None:
        if storage_type == "sqlite":
            _storage_instance = SQLiteBudgetStorage(**kwargs)
        else:
            _storage_instance = SimpleBudgetStorage(**kwargs)
    
    return _storage_instance