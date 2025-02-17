from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from ipdb import set_trace as st

# Initialize WebDriver (ensure you have the appropriate driver installed, e.g., ChromeDriver)
driver = webdriver.Chrome()

try:
    # Open Mixamo website
    driver.get("https://www.mixamo.com/")

    # Log in manually or automate login
    print("Log in to Mixamo and then press Enter here...")
    input()  # Wait for manual login
    
    characters_tab = driver.find_elements(By.LINK_TEXT, "Characters")[0]  # Update with actual class or selector
    print(characters_tab)
    characters_tab.click()

    # tags = driver.find_element(By.CLASS_NAME, 'pagination-holder').find_elements(By.TAG_NAME, 'a')[:-1]
    # for tag in tags:
    #     try:
    #         tag.click()
    #         time.sleep(2)
    #     except:
    #         print('')
    # Example: Loop through characters (you'll need to inspect Mixamo's structure)
    #site > div:nth-child(5) > div > div > div.product-results-holder.col-sm-6 > div > div.product-results > div > div:nth-child(2)
    characters = driver.find_element(By.CLASS_NAME, 'product-results').find_elements(By.CLASS_NAME, "product-overlay")  # Update with actual class or selector
    print(characters)
    for index, character in enumerate(characters[9:]):
        # Click on each character
        character.click()
        time.sleep(2)
        confirm_button = driver.find_element(By.CLASS_NAME, "modal-footer").find_element(By.XPATH,'button')
        confirm_button.click()
        time.sleep(5)
        # Click the download button (inspect the element for accuracy)
        download_button = driver.find_element(By.CLASS_NAME, "product-preview").find_element(By.CLASS_NAME,'sidebar-header').find_element(By.CSS_SELECTOR, 'button.btn-block.btn.btn-primary')  # Update with actual ID or selector
        download_button.click()
        # body > div:nth-child(9) > div > div.asset-download-modal.fade.in.modal > div > div > div.modal-footer > div > button.btn.btn-primary
        download_confirm_button = driver.find_element(By.CLASS_NAME, "modal-footer").find_element(By.CSS_SELECTOR,'button.btn.btn-primary')
        download_confirm_button.click()
        #site > div:nth-child(5) > div > div > div.product-preview-holder.col-sm-6 > div > div.editor.row.row-no-gutter > div.editor-sidebar.col-xs-4 > div.sidebar-header > button.btn-block.btn.btn-primary
        print(f"Downloaded character {index + 1}")
        time.sleep(10)  # Adjust for download time
finally:
    driver.quit()
