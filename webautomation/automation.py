from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import winsound

def refresh():
    time.sleep(1)
    driver.refresh()

driver = webdriver.Chrome(ChromeDriverManager().install())

# Login
driver.get("https://hk.sz.gov.cn:8118/")
select = Select(WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "select_certificate"))))
select.select_by_visible_text('往來港澳通行證')
id_input = driver.find_element_by_id("input_idCardNo")
id_input.send_keys('C63103428')
passwd = driver.find_element_by_id("input_pwd")
passwd.send_keys('lck215069')
input = driver.find_element_by_xpath('//*[@id="input_verifyCode"]')
input.click()
#
# verifycode = driver.find_element_by_id("input_verifyCode")
# verifycode.send_keys("")

element1 = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.ID, 'a_canBookHotel')))
element1.click()

while True:
    try:
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button[text()='預約']")))
        buttons = driver.find_elements_by_xpath("//*[contains(text(), '預約') and @class='orange']")
        if len(buttons) == 0:
            refresh()
            continue
        for idx, b in enumerate(buttons):
            print('gogo')
            if idx + 1 in [5, 6, 7]:
                b.click()
                confirm = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[text()='確定']")))
                confirm.click()
                winsound.Beep(840, 500)
                print("Gogogo~")
                verify_box = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "checkCode")))
                verify_box.click()
                img_verify = driver.find_element_by_id("img_verify")
            else:
                continue

    except (TimeoutException, NoSuchElementException):
        refresh()
        continue


'<img id="img_verify" src="/user/getVerify?0.486996814105884" onclick="getVerify(this);">'