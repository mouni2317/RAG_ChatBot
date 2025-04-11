from kafka import KafkaConsumer
from app.DBServices.db_factory import get_db_writer  # factory method for DB writers
import logging
import random

logging.basicConfig(level=logging.INFO)

class DBWriteService:
    def __init__(self, db_type="mongo"):
        self.db_writer = get_db_writer(db_type)

    def validate_and_transform(self, event):
        # Basic validation and transformation logic
        if not isinstance(event, dict):
            return None

        if "title" not in event:
            return None

        # Example transformation: ensure keys are lowercased
        transformed = {k.lower(): v for k, v in event.items()}
        return transformed
        
    def process_event(self, event):
        # Validate or transform if needed
        if isinstance(event, dict) and "title" in event:
            #inserted_id = self.db_writer.write(event)
            inserted_id = random.randint(1, 100)  # Simulate DB write
            if inserted_id:
                logging.info(f"✅ Inserted: {inserted_id}")
            else:
                logging.warning("⚠️ Duplicate or write failed")
        else:
            logging.error("❌ Invalid event format")

    def run_with_consumer(self):
        consumer = KafkaConsumer("crawler_events")
        logging.info("🚀 DB Write Service started with Kafka consumer...")
        for msg in consumer:
            self.process_event(msg.value)
