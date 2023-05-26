from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
import time
import yaml


def get_website(url: str) -> webdriver.Chrome:
    """
    Obtém o website usando o Chrome driver.

    Args:
        url (str): A URL do website a ser obtido.

    Returns:
        webdriver.Chrome: O objeto driver que representa o website.
    """
    driver = webdriver.Chrome()
    driver.get(url)
    return driver


def get_login(url: str, login: str, password: str) -> webdriver.Chrome:
    """
    Realiza o login usando e-mail e senha.

    Args:
        url (str): A URL da página de login.
        login (str): O endereço de e-mail do usuário.
        password (str): A senha do usuário.

    Returns:
        webdriver.Chrome: O objeto driver que representa o website logado.
    """
    driver = get_website(url)
    email = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "input[name='text']")))
    email.send_keys(login)
    driver.find_element(By.XPATH, "//span[text()='Avançar']").click()
    if EC.text_to_be_present_in_element((By.XPATH, "//span[contains(text(), 'Houve um acesso incomum à sua conta')]"),
                                        'Houve um acesso incomum à sua conta. Para ajudar a mantê-la protegida, insira seu número de celular ou endereço de e-mail para confirmar que é você.'):
        email = WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "input[name='text']")))
        email.send_keys(login[:-10])
        driver.find_element(By.XPATH, "//span[text()='Avançar']").click()
    password_input = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "input[name='password']")))
    password_input.send_keys(password)
    password_input.send_keys(Keys.RETURN)
    WebDriverWait(driver, 10).until(EC.url_changes(url))
    return driver


def process(config: dict, search_query: list[str]):
    """
    Pesquisa os tweets fornecidos.

    Args:
        config (dict): O dicionário que contém os valores de configuração.
    """
    data = set()
    data_dict = {}
    scrolls = 1
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    for search_line in search_query:
        URL = f'https://twitter.com/search?q={search_line}&src=typed_query'
        try:
            driver = get_login(
                config["url"], config["login"], config["password"])
            time.sleep(2)
            driver.get(URL)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, '[data-testid="tweetText"]')))
        except WebDriverException:
            print("Tweets did not appear! Proceeding after timeout")
        wait = WebDriverWait(driver, 10)
        while scrolls < 10:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            scrolls += 1
        tweets = wait.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, '[data-testid="tweet"]')))
        for tweet in tweets:
            try:
                tweet_text = WebDriverWait(tweet, 10).until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'div[data-testid="tweetText"]'))).text
                data.add(tweet_text.replace('\n', ' '))
            except WebDriverException:
                pass
        data_dict[search_line] = data
    print(data_dict)
    driver.quit()


if __name__ == "__main__":
    with open("config.yml", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    process(config, ["BBB"])
