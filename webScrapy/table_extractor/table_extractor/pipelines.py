import csv
import os
from itemadapter import ItemAdapter

class TableExtractorPipeline:
    def __init__(self):
        self.file = None
        self.writer = None
        self.file_exists = os.path.exists('agmarknet_data.csv')

    def open_spider(self, spider):
        # Open CSV file in append mode
        self.file = open('agmarknet_data.csv', 'a', newline='', encoding='utf-8')
        fieldnames = [
            'sl_no', 'district', 'market', 'commodity', 'variety',
            'grade', 'min_price', 'max_price', 'modal_price', 'price_date'
        ]
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        # Only write the header if the file is new
        if not self.file_exists:
            self.writer.writeheader()

    def process_item(self, item, spider):
        # self.writer.writerow(ItemAdapter(item).asdict())
        return item

    def close_spider(self, spider):
        if self.file:
            self.file.close()
