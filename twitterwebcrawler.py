from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver
import logging
import time


class TwitterScraper:
    def __init__(self, config_file: dict, logger: logging.Logger) -> None:
        self.config = config_file
        self.driver = None
        self.logger = logger

    def get_website(self, url: str):
        try:
            self.logger.info(f"Getting website: {url}")
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            return driver
        except Exception as e:
            self.logger.error(
                f"Error occurred while getting website: {str(e)}")
            exit()

    def get_login(self, url: str, login: str, password: str):
        try:
            self.logger.info("Logging in website...")
            driver = self.get_website(url)
            self._wait_and_search(
                "Entering email...", driver, "input[name='text']", login
            )
            if EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="ocfEnterTextTextInput"]')):
                self.logger.info("Handling unusual access...")
                email = WebDriverWait(driver, 5).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, "input[name='text']")))
                email.send_keys(login[:-10])
                email.send_keys(Keys.RETURN)
            self._wait_and_search(
                "Entering password...", driver, "input[name='password']", password
            )
            time.sleep(2)
            if EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0RichTextInputContainer"]')):
                self.logger.info("Website login successful.")
                return driver
            else:
                raise BaseException("Something Wrong")
        except Exception as e:
            self.logger.error(f"Error occurred during website login: {str(e)}")
            exit()

    # TODO Rename this here and in `get_login`
    def _wait_and_search(self, arg0, driver, arg2, arg3):
        self.logger.info(arg0)
        email = WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, arg2))
        )
        email.send_keys(arg3)
        email.send_keys(Keys.RETURN)

    def get_tweets(self, search_query: list[str]) -> dict[str, set[str]]:
        try:
            self.logger.info(
                f"Starting tweet collection for {len(search_query)} search queries.")
            data_dict = {}
            self.driver = self.get_login(
                self.config["url"], self.config["login"], self.config["password"])
            for search_line in search_query:
                data = set()
                URL = f'https://twitter.com/search?q={search_line}&src=typed_query'
                self.logger.info(
                    f"Searching for {search_line} ({search_query.index(search_line)+1}/{len(search_query)})...")
                try:
                    self.driver.get(URL)
                    time.sleep(3)
                    WebDriverWait(self.driver, 10).until(EC.presence_of_element_located(
                        (By.CSS_SELECTOR, '[data-testid="tweetText"]')))
                except WebDriverException:
                    self.logger.error(
                        f"Tweets didn't appear! Report 1: {self.driver.find_element(By.TAG_NAME, 'body').text}")
                    exit()
                wait = WebDriverWait(self.driver, 10)
                self.logger.info("Collecting tweets...")
                time.sleep(1)
                for _ in range(1, 10):
                    self.driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                try:
                    tweets = wait.until(EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, '[data-testid="tweet"]')))
                except WebDriverException:
                    self.logger.error(
                        f"Tweets didn't appear! Report 2: {self.driver.find_element(By.TAG_NAME, 'body').text}")
                    exit()
                for tweet in tweets:
                    try:
                        tweet_text = WebDriverWait(tweet, 10).until(EC.presence_of_element_located(
                            (By.CSS_SELECTOR, 'div[data-testid="tweetText"]'))).text
                        data.add(tweet_text.replace('\n', ' '))
                    except WebDriverException as e:
                        self.logger.error(
                            f"Error occurred while collecting tweet: {str(e)}")
                data_dict[search_line] = data
            self.driver.quit()
            self.logger.info("Tweet collection completed.")
            return data_dict
        except Exception as e:
            self.logger.error(f"Error occurred while getting tweets: {str(e)}")
            exit()
