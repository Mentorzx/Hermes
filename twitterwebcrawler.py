from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re


def get_website(url: str):
    driver = webdriver.Chrome()
    driver.get(url)
    return driver


def get_login(url: str, login: str, password: str):
    driver = get_website(url)
    email = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "input[name='text']")))
    email.send_keys(login)
    driver.find_element(By.XPATH, "//span[text()='Avançar']").click()
    if EC.text_to_be_present_in_element((By.XPATH, "//span[contains(text(), 'Houve um acesso incomum à sua conta')]"), 'Houve um acesso incomum à sua conta. Para ajudar a mantê-la protegida, insira seu número de celular ou endereço de e-mail para confirmar que é você.'):
        email = WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "input[name='text']")))
        email.send_keys(login[:-10])
        driver.find_element(By.XPATH, "//span[text()='Avançar']").click()
    senha = WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, "input[name='password']")))
    senha.send_keys(password)
    senha.send_keys(Keys.RETURN)
    WebDriverWait(driver, 10).until(EC.url_changes(url))
    return driver


def process(url: str, login: str, password: str):
    pattern1 = r".( mil)? Tweets"
    pattern2 = r".+ · Assunto do Momento"
    blacklist = ['Tendências de Brasil', 'Para você', 'Assuntos do Momento',
                 'Notícias', 'Esportes', 'Entretenimento', 'Pop', '·']
    trendings = set()
    driver = get_login(url, login, password)
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, 'span.css-16my406')))
    driver.get('https://twitter.com/explore/tabs/trending')
    trends = driver.find_elements(By.CSS_SELECTOR, 'span.css-16my406')
    for trend in trends:
        name = trend.text
        match1 = re.search(pattern1, name)
        match2 = re.search(pattern2, name)
        if name in blacklist or not name or name.isnumeric() or match1 or match2:
            continue
        trendings.add(name)
    print(trendings)
    driver.quit()


if __name__ == "__main__":
    url = 'https://twitter.com/login'
    login = 'iasearch333@gmail.com'
    password = 'iasearch1234'

    process(url, login, password)
