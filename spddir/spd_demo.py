import scrapy
from imagecrawler.items import ImageItem

curr = 1
end = 0
AMOUNT = 15

class pexelsScraper(scrapy.Spider):
    name = "spd_demo"

    def start_requests(self):
        url = "http://rusdemotivator.ru"
        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        image = ImageItem()
        img_url = []
        global curr
        global end
        global AMOUNT

        #img_url = response.css('[id="demo_image"]::attr(src)').extract_first()
        #img_url = response.css('[id="orating-67822"] img::attr(src)').extract_first()

        for it in range(AMOUNT):
            img_url.append(self.full_url(response.css('td.newsstory img::attr(src)')[it].extract()))
        image["image_urls"] = img_url
        if curr == 1:
            end = self.atoi(response.css('div.navigation a::text')[9].extract())
          #  prt = end
        #else:
         #   prt = self.atoi(response.css('div.navigation a::text')[10].extract())
        if curr < end:
            if curr > 1:
                nav = response.css('div.navigation a::attr(href)')[11].extract()
            else:
                nav = response.css('div.navigation a::attr(href)')[10].extract()
            curr += 1
            yield scrapy.Request(nav, callback=self.parse)
        yield image

    def full_url(self, string):
        url = 'http://rusdemotivator.ru' + string
        return url

    def atoi(self, string):
        base = ord('0')
        num = 0
        for i in range(len(string)):
            num = num*10 + ord(string[i])-base
        return num



