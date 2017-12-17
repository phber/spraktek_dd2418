# encoding=utf8

import scrapy 
import json
import dateutil.parser as dt
import datetime
import re

BOSTADSRATT_URL = 'https://www.hemnet.se/bostader?item_types%5B%5D=bostadsratt'
BOSTADSRATT_URLS = [BOSTADSRATT_URL + '&page=' + str(i) for i in range(1,186)]

class HemnetSpider(scrapy.Spider):
    name = "spider"

    def start_requests(self):
        urls = BOSTADSRATT_URLS
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        hrefs = response.css('a.item-link-container::attr(href)').extract()
        hrefs = hrefs[5:] # Remove maklartips
        for u in hrefs:
            h_url = 'https://www.hemnet.se' + u
            yield scrapy.Request(h_url, callback=self.parse_ad)

    def parse_ad(self, response):
        lan = response.css('.breadcrumbs__link::text').extract_first() 
        if 'Stockholm' in lan:
            return
        visits = response.css('.property-stats__visits::text').extract_first()
        if visits != None: 
            visits = ''.join(visits.split())
        else:
            visits = '0'
        data = response.xpath('//script/text()').extract()
        date = response.css('span::attr(datetime)').extract_first() 
        if date is None:
            return
        d = dt.parse(date)
        published = datetime.date(d.year, d.month, d.day)
        days = (datetime.date.today() - published).days
        for tag in data:
            if '"@type": "Product"' in tag:
                fixed = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'', tag)
                d = json.loads(fixed)
                if 'description' not in d or 'offers' not in d:
                    return
                descr = d['description'].encode('utf-8')
                price = d['offers']['price']
                if int(price) < 500000:
                    return
                break
        yield {
            'description': descr,
            'price': int(price),
            'visits': int(visits),
            'days' : days,
            'published' : published
        }
