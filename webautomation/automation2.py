from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import winsound
from itertools import cycle


def try_and_refresh(driver, operations):
    if not isinstance(operations, (list, tuple)):
        operations = (operations,)
    while True:
        try:
            for op in operations:
                op(driver)
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


def login_stage1(driver):
    my_account = driver.find_element_by_xpath('/html/body/div/div/div[6]/div[2]/div[2]')
    my_account.click()
    login_icon = driver.find_element_by_xpath('/html/body/div/div/div[1]/div/div/p')
    login_icon.click()


def login_stage2(driver):
    email_input = driver.find_element_by_xpath('/html/body/div/div/div[2]/div/form/div/div[1]/input')
    email_input.send_keys('luchongkai96@gmail.com')
    passwd_input = driver.find_element_by_xpath('/html/body/div/div/div[2]/div/form/div/div[3]/input')
    passwd_input.send_keys('yourpasswd')
    remember_icon = driver.find_element_by_xpath('/html/body/div/div/div[2]/div/form/div/div[6]/div')
    remember_icon.click()
    login_icon = driver.find_element_by_xpath('/html/body/div/div/div[2]/div/form/div/div[8]')
    login_icon.click()


def click_to_zhuhai(driver):
    time.sleep(2)
    to_zhuhai = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '/html/body/div/div/div[3]/table/tr[1]/td[1]/p')))
    driver.execute_script("arguments[0].click();", to_zhuhai)


def input_particulars(driver):
    name_input = driver.find_element_by_xpath('//*[@id="app"]/div/div[6]/div/div[2]/div[1]/div[2]/input')
    name_input.send_keys('LU Chongkai')
    passport_input = driver.find_element_by_xpath('//*[@id="app"]/div/div[6]/div/div[2]/div[2]/div[2]/input')
    passport_input.send_keys('your info')
    agree_icon = driver.find_element_by_xpath('//*[@id="app"]/div/div[9]/span[1]')
    driver.execute_script("arguments[0].click();", agree_icon)


def check_date(driver):
    for i in cycle(range(18, 25)):  # 18 = 1.13
        icon = driver.find_element_by_xpath('/html/body/div/div/div[2]/div[1]/div[1]/span[2]')
        driver.execute_script("arguments[0].click();", icon)
        date = driver.find_element_by_xpath(f'/html/body/div/div/div[2]/div[3]/section/div/div[3]/div[{i}]/div')
        driver.execute_script("arguments[0].click();", date)
        preorder_icon = driver.find_element_by_xpath('//*[@id="app"]/div/div[2]/div[1]/div[3]/div[2]/span')
        driver.execute_script("arguments[0].click();", preorder_icon)
        if driver.find_elements_by_class_name('picker-selected'):
            confirm_date = driver.find_element_by_xpath('//*[@id="app"]/div/div[2]/div[5]/div[1]/span[2]')
            driver.execute_script("arguments[0].click();", confirm_date)
            confirm_purchase = driver.find_element_by_xpath('//*[@id="app"]/div/div[10]')
            driver.execute_script("arguments[0].click();", confirm_purchase)
            # fail = driver.find_elements_by_class_name('yd-toast-content')
            # if fail:
            #     continue
            time.sleep(1)
            if driver.current_url != 'https://i.hzmbus.com/webhtml/details_payment':
                print(f'Find one ticket on 01.{i - 5} but token by others')
                continue
            driver.switch_to.window(driver.current_window_handle)
            success = input("Successfully get it? if no then continue")
            if success == 'yes' or success == 'y':
                print('Finished')
                break
            else:
                continue
        else:
            continue


def main():
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get('https://i.hzmbus.com/webhtml/index.html')
    try_and_refresh(driver, [login_stage1, login_stage2, click_to_zhuhai, input_particulars, check_date])

main()
