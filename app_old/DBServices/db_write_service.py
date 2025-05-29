from app_old.DBServices.db_factory import get_db_writer  # factory method for DB writers
import logging
import random
from app_old.DBServices.chroma_writer import ChromaWriter

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
        if isinstance(event, dict):
            if self.db_writer:
                if isinstance(self.db_writer, ChromaWriter):
                    # For Chroma, assume event contains 'texts' and optional 'metadatas'
                    texts = event.get('texts', [])
                    metadatas = event.get('metadatas', None)
                    ids = event.get('ids', [])
                    self.db_writer.write(texts, ids, metadatas)
                    logging.info("✅ Chroma DB write successful")
                else:
                    # For MongoDB and others, assume event is a single document
                    inserted_id = self.db_writer.write(event)
                    if inserted_id:
                        logging.info(f"✅ Inserted: {inserted_id}")
                    else:
                        logging.warning("⚠️ Duplicate or write failed")
            else:
                logging.error("❌ No DB writer available")
        else:
            logging.error("❌ Invalid event format")

    # def run_with_consumer(self):
    #     consumer = KafkaConsumer("crawler_events")
    #     logging.info("🚀 DB Write Service started with Kafka consumer...")
    #     for msg in consumer:
    #         self.process_event(msg.value)
