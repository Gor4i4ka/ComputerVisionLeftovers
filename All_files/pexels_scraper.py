import scrapy
from imagecrawler.items import ImageItem

AMOUNT = 20

class pexelsScraper(scrapy.Spider):
    name = "spd_mem"

    def start_requests(self):
        url = "http://1001mem.ru/best/5"
        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        image = ImageItem()
        img_url = []
        global AMOUNT

        #img_url = response.css('[id="demo_image"]::attr(src)').extract_first()
        #img_url = response.css('[id="orating-67822"] img::attr(src)').extract_first()

        for it in range(AMOUNT):
            img_url.append(response.css('div.image img::attr(src)')[it].extract())
        image["image_urls"] = img_url
        nav = response.css('div.pagination a::attr(href)')[1].extract()
       # nav = self.full_url(nav)
        #yield scrapy.Request(nav, callback=self.parse)
        yield response.follow(nav, callback = self.parse)
        yield image

    def full_url(self, string):
        url = 'http://1001mem.ru' + string
        return url

    def atoi(self, string):
        base = ord('0')
        num = 0
        for i in range(len(string)):
            num = num*10 + ord(string[i])-base
        return num


