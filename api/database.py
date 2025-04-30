import os
import logging
import motor.motor_asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from bson import ObjectId

# Load environment variables
load_dotenv()

class MongoDB:
    """MongoDB connection and utility functions for the financial database."""
    
    # Class variables for connection
    client = None
    db = None
    
    @classmethod
    async def connect(cls):
        """
        Establish a connection to MongoDB.
        Uses the MONGODB_URI environment variable for connection.
        """
        try:
            mongodb_uri = os.getenv("MONGODB_URI")
            if not mongodb_uri:
                logging.error("MONGODB_URI environment variable not set")
                return False
            
            # Create a new client and connect to the server
            cls.client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_uri)
            # Select the database
            db_name = os.getenv("MONGODB_DB", "financial_db")
            cls.db = cls.client[db_name]
            
            # Ping the database to verify connection
            await cls.client.admin.command("ping")
            logging.info(f"Successfully connected to MongoDB database: {db_name}")
            return True
        except Exception as e:
            logging.error(f"MongoDB connection error: {str(e)}")
            cls.client = None
            cls.db = None
            return False
    
    @classmethod
    async def close(cls):
        """Close the MongoDB connection."""
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None
            logging.info("MongoDB connection closed")
    
    @classmethod
    async def get_contacts(cls, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all contacts for a user.
        
        Args:
            user_id: The user ID to get contacts for
            
        Returns:
            List of contact documents
        """
        try:
            if not cls.db:
                await cls.connect()
            
            # Use user_id to filter contacts
            cursor = cls.db.contacts.find({"userId": user_id})
            contacts = await cursor.to_list(length=None)
            
            # Convert ObjectId to string for JSON serialization
            for contact in contacts:
                if "_id" in contact and isinstance(contact["_id"], ObjectId):
                    contact["_id"] = str(contact["_id"])
            
            return contacts
        except Exception as e:
            logging.error(f"Error retrieving contacts: {str(e)}")
            return []
    
    @classmethod
    async def get_expenses(cls, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all expenses for a user.
        
        Args:
            user_id: The user ID to get expenses for
            
        Returns:
            List of expense documents
        """
        try:
            if not cls.db:
                await cls.connect()
            
            # Use user_id to filter expenses
            cursor = cls.db.expenses.find({"userId": user_id})
            expenses = await cursor.to_list(length=None)
            
            # Convert ObjectId to string for JSON serialization
            for expense in expenses:
                if "_id" in expense and isinstance(expense["_id"], ObjectId):
                    expense["_id"] = str(expense["_id"])
            
            return expenses
        except Exception as e:
            logging.error(f"Error retrieving expenses: {str(e)}")
            return []
    
    @classmethod
    async def get_products(cls, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all products for a user.
        
        Args:
            user_id: The user ID to get products for
            
        Returns:
            List of product documents
        """
        try:
            if not cls.db:
                await cls.connect()
            
            # Use user_id to filter products
            cursor = cls.db.products.find({"userId": user_id})
            products = await cursor.to_list(length=None)
            
            # Convert ObjectId to string for JSON serialization
            for product in products:
                if "_id" in product and isinstance(product["_id"], ObjectId):
                    product["_id"] = str(product["_id"])
            
            return products
        except Exception as e:
            logging.error(f"Error retrieving products: {str(e)}")
            return []
    
    @classmethod
    async def get_transactions(cls, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all transactions for a user.
        
        Args:
            user_id: The user ID to get transactions for
            
        Returns:
            List of transaction documents
        """
        try:
            if not cls.db:
                await cls.connect()
            
            # Use user_id to filter transactions
            cursor = cls.db.transactions.find({"userId": user_id})
            transactions = await cursor.to_list(length=None)
            
            # Convert ObjectId to string for JSON serialization
            for transaction in transactions:
                if "_id" in transaction and isinstance(transaction["_id"], ObjectId):
                    transaction["_id"] = str(transaction["_id"])
            
            return transactions
        except Exception as e:
            logging.error(f"Error retrieving transactions: {str(e)}")
            return [] 