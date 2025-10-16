from playwright.sync_api import sync_playwright, Page, expect

def run_verification(page: Page):
    """
    This script verifies the new 'Save Changes' functionality in the settings panel.
    """
    # 1. Navigate to the dashboard
    page.goto("http://localhost:8080/ui")

    # 2. Log in
    # The login modal should be visible by default
    # We'll use a dummy key since the backend doesn't validate it in this test environment
    api_key_input = page.get_by_label("API Key")
    expect(api_key_input).to_be_visible()
    api_key_input.fill("dummy-key")
    page.get_by_role("button", name="Sign in").click()

    # Wait for the main content to load by checking for the settings panel
    settings_header = page.get_by_role("heading", name="System Settings")
    expect(settings_header).to_be_visible()

    # 3. Change a setting
    # Let's change the logging level
    log_level_selector = page.get_by_label("Level")
    expect(log_level_selector).to_be_visible()
    log_level_selector.select_option("DEBUG")

    # 4. Verify the "Save Changes" button is enabled
    save_button = page.get_by_role("button", name="Save Changes")
    expect(save_button).to_be_enabled()

    # 5. Click the "Save Changes" button
    save_button.click()

    # 6. Verify the "Save Changes" button is disabled again after saving
    expect(save_button).to_be_disabled()

    # 7. Take a screenshot
    page.screenshot(path="docs/archive/jules-verification/verification.png")

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        run_verification(page)
        browser.close()

if __name__ == "__main__":
    main()