import scrapy
from table_extractor.items import TableExtractorItem
from datetime import datetime

today = datetime.today().strftime("%d-%b-%Y")
# today = "08-May-2025"
class TableSpider(scrapy.Spider):
    name = 'table_spider'
    
    start_urls = [ f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity=2&Tx_State=TL&Tx_District=0&Tx_Market=0&DateFrom={today}&DateTo={today}&Fr_Date={today}&To_Date={today}&Tx_Trend=0&Tx_CommodityHead=Paddy(Dhan)(Common)&Tx_StateHead=Telangana&Tx_DistrictHead=--Select--&Tx_MarketHead=--Select--"]
    
    

    def parse(self, response):
        table = response.css('table.tableagmark_new')
        
        for row in table.css('tr')[1:]:  # Skip header row
            item = TableExtractorItem()
            row_data = []
            
            for td in row.css('td'):
                span_text = td.css('span::text').get()
                if span_text:
                    row_data.append(span_text.strip())
                else:
                    row_data.append(' '.join(td.css('::text').getall()).strip())
            
            if row_data:  # Only process non-empty rows
                # print(row_data)
                item['sl_no'] = row_data[0]
                item['district'] = row_data[1]
                item['market'] = row_data[2]
                item['commodity'] = row_data[3]
                item['variety'] = row_data[4]
                item['grade'] = row_data[5]
                item['min_price'] = row_data[6]
                item['max_price'] = row_data[7]
                item['modal_price'] = row_data[8]
                item['price_date'] = row_data[9]
                
                # print(row_data)
                yield item