# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class TableExtractorItem(scrapy.Item):
    # define the fields for your item here like:
    sl_no = scrapy.Field()
    district = scrapy.Field()
    market = scrapy.Field()
    commodity = scrapy.Field()
    variety = scrapy.Field()
    grade = scrapy.Field()
    min_price = scrapy.Field()
    max_price = scrapy.Field()
    modal_price = scrapy.Field()
    price_date =  scrapy.Field()
    
