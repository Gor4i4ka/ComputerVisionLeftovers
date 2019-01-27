import scrapy
from imagecrawler.items import ImageItem


class pexelsScraper(scrapy.Spider):
    name = "pexels"

    def start_requests(self):
        #url = "https://www.demotivation.us/newest/all/women-wont-date-a-guy-that-still-lives-with-his-1289418.html"
        url = "http://rusdemotivator.ru/demotivatory-o-zhizni/"
        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        image = ImageItem()
        #img_url = response.css('[id="demo_image"]::attr(src)').extract_first()
        #img_url = response.css('[id="orating-67822"] img::attr(src)').extract_first()
        img_url = response.css('td.newsstory img::attr(src)').extract_first()
        print('SUKAAAAAAAAAAAAAAAAAAAA')
        print(self.full_url(img_url))
        image["image_urls"] = self.full_url(img_url)
        yield {'image_urls': self.full_url(img_url) }
        #yield image

    def full_url(self, string):
        url_list = []
        url_list.append('http://rusdemotivator.ru' + string)
        return url_list


