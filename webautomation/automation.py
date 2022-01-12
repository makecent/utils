from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import winsound


def try_and_refresh(driver, operation, return_value=False, ):
    while True:
        try:
            if return_value:
                return operation(driver)
            else:
                operation(driver)
                break
        except (TimeoutException, NoSuchElementException, ValueError) as e:
            while True:
                try:
                    time.sleep(0.1)
                    driver.refresh()
                    break
                except TimeoutException:
                    continue
            continue


def input_particulars(driver):
    select = Select(
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "select_certificate"))))
    select.select_by_visible_text('往來港澳通行證')
    id_input = driver.find_element_by_id("input_idCardNo")
    id_input.send_keys('C63103428')
    passwd = driver.find_element_by_id("input_pwd")
    passwd.send_keys('lck162599')
    input = driver.find_element_by_xpath('//*[@id="input_verifyCode"]')
    input.click()
    winsound.Beep(400, 500)
    # verifycode = driver.find_element_by_id("input_verifyCode")
    # verifycode.send_keys("")


def make_appointment(driver):
    element = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.ID, 'a_canBookHotel')))
    element.click()


def get_tickets(driver):
    if driver.current_url == 'https://hk.sz.gov.cn:8118/userPage/login':
        raise TypeError
    WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "//button[text()='預約']")))
    buttons = driver.find_elements_by_xpath("//*[contains(text(), '預約') and @class='orange']")
    if len(buttons) == 0:
        raise ValueError
    for idx, b in enumerate(buttons):
        date = b.get_attribute('onclick').split("'")[1]
        if date not in ['2022-01-12', '2022-01-13', '2022-01-14', '2022-01-15', '2022-01-16', '2022-01-17', '2022-01-18']:
            continue
        b.click()
        confirm = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='確定']")))
        confirm.click()
        winsound.Beep(840, 500)
        verify_box = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "checkCode")))
        verify_box.click()
        winsound.Beep(840, 500)
        driver.switch_to.window(driver.current_window_handle)
        print('gogogo')
    raise ValueError


def main():
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get("https://hk.sz.gov.cn:8118/")
    while True:
        try_and_refresh(driver, input_particulars)
        try_and_refresh(driver, make_appointment)
        try:
            try_and_refresh(driver, get_tickets)
            break
        except TypeError:
            continue
main()