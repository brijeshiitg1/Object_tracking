
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

class MongoDatabase:
    """
    MongoDB handler for storing object tracking events and analytics.
    """
    
    def __init__(self, uri: str = "mongodb://localhost:27017/", 
                 db_name: str = "object_tracking", 
                 collection_name: str = "traffic_events"):
        """
        Initialize MongoDB connection.
        
        Args:
            uri: MongoDB connection URI
            db_name: Database name
            collection_name: Collection name for storing events
        """
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        try:
            # Connect to MongoDB
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.logger.info(f"Connected to MongoDB at {uri}")
            
            # Get database and collection
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            
            # Create indexes for better query performance
            self._create_indexes()
            
        except ConnectionFailure as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes on commonly queried fields."""
        try:
            self.collection.create_index("object_id")
            self.collection.create_index("timestamp")
            self.collection.create_index("object_name")
            self.collection.create_index([("object_id", 1), ("timestamp", -1)])
            self.logger.info("Indexes created successfully")
        except OperationFailure as e:
            self.logger.warning(f"Could not create indexes: {e}")
    
    def insert_record(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Insert a single tracking event record.
        
        Args:
            record: Dictionary containing event data
            
        Returns:
            Inserted document ID or None if failed
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in record or not isinstance(record['timestamp'], datetime):
                record['created_at'] = datetime.now()
            else:
                # Convert CV2 tick count to datetime if needed
                record['created_at'] = datetime.now()
                record['tick_count'] = record.get('timestamp')
            
            result = self.collection.insert_one(record)
            self.logger.debug(f"Inserted record with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"Error inserting record: {e}")
            return None
    
    def insert_many_records(self, records: List[Dict[str, Any]]) -> Optional[List[str]]:
        """
        Insert multiple tracking event records.
        
        Args:
            records: List of dictionaries containing event data
            
        Returns:
            List of inserted document IDs or None if failed
        """
        try:
            # Add timestamps
            for record in records:
                if 'created_at' not in record:
                    record['created_at'] = datetime.now()
            
            result = self.collection.insert_many(records)
            self.logger.info(f"Inserted {len(result.inserted_ids)} records")
            return [str(id) for id in result.inserted_ids]
            
        except Exception as e:
            self.logger.error(f"Error inserting multiple records: {e}")
            return None
    
    def find_by_object_id(self, object_id: int, limit: int = 100) -> List[Dict]:
        """
        Find all events for a specific object ID.
        
        Args:
            object_id: The tracking object ID
            limit: Maximum number of records to return
            
        Returns:
            List of matching documents
        """
        try:
            cursor = self.collection.find(
                {"object_id": object_id}
            ).sort("created_at", -1).limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error finding records by object_id: {e}")
            return []
    
    def find_by_object_name(self, object_name: str, limit: int = 100) -> List[Dict]:
        """
        Find all events for a specific object type/class.
        
        Args:
            object_name: The object class name (e.g., 'car', 'person')
            limit: Maximum number of records to return
            
        Returns:
            List of matching documents
        """
        try:
            cursor = self.collection.find(
                {"object_name": object_name}
            ).sort("created_at", -1).limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error finding records by object_name: {e}")
            return []
    
    def find_by_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """
        Find all events within a time range.
        
        Args:
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            List of matching documents
        """
        try:
            cursor = self.collection.find({
                "created_at": {
                    "$gte": start_time,
                    "$lte": end_time
                }
            }).sort("created_at", 1)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error finding records by time range: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics from the database.
        
        Returns:
            Dictionary containing statistics
        """
        try:
            total_events = self.collection.count_documents({})
            
            # Count by object type
            pipeline = [
                {"$group": {
                    "_id": "$object_name",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
            object_counts = list(self.collection.aggregate(pipeline))
            
            # Count unique objects
            unique_objects = len(self.collection.distinct("object_id"))
            
            return {
                "total_events": total_events,
                "unique_objects": unique_objects,
                "object_type_counts": object_counts
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    def get_object_frequency(self, hours: int = 24) -> List[Dict]:
        """
        Get frequency of object detections in the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of object frequencies
        """
        try:
            time_threshold = datetime.now() - datetime.timedelta(hours=hours)
            
            pipeline = [
                {"$match": {
                    "created_at": {"$gte": time_threshold}
                }},
                {"$group": {
                    "_id": "$object_name",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
            
            return list(self.collection.aggregate(pipeline))
            
        except Exception as e:
            self.logger.error(f"Error getting object frequency: {e}")
            return []
    
    def delete_old_records(self, days: int = 30) -> int:
        """
        Delete records older than specified days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of deleted documents
        """
        try:
            time_threshold = datetime.now() - datetime.timedelta(days=days)
            
            result = self.collection.delete_many({
                "created_at": {"$lt": time_threshold}
            })
            
            self.logger.info(f"Deleted {result.deleted_count} old records")
            return result.deleted_count
            
        except Exception as e:
            self.logger.error(f"Error deleting old records: {e}")
            return 0
    
    def clear_collection(self) -> bool:
        """
        Clear all records from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.collection.delete_many({})
            self.logger.warning(f"Cleared {result.deleted_count} records from collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
            return False
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.logger.info("MongoDB connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage and testing
if __name__ == "__main__":
    # Initialize database
    db = MongoDatabase(
        uri="mongodb://localhost:27017/",
        db_name="object_tracking",
        collection_name="traffic_events"
    )
    
    # Insert sample record
    sample_record = {
        "object_id": 1,
        "object_name": "car",
        "timestamp": datetime.now(),
        "aoi_name": "Restricted Zone",
        "event_type": "entry"
    }
    
    record_id = db.insert_record(sample_record)
    print(f"Inserted record: {record_id}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"Database statistics: {stats}")
    
    # Find records by object ID
    records = db.find_by_object_id(1)
    print(f"Found {len(records)} records for object ID 1")
    
    # Close connection
    db.close()