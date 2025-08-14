import pdb
import asyncio
import pyperclip
from typing import Optional, Type
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller, DoneAction
from main_content_extractor import MainContentExtractor
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
import logging

logger = logging.getLogger(__name__)


class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()

    # def _register_custom_actions(self):
    #     """Register all custom browser actions"""

    #     @self.registry.action("Copy text to clipboard")
    #     def copy_to_clipboard(text: str):
    #         pyperclip.copy(text)
    #         return ActionResult(extracted_content=text)

    #     @self.registry.action("Paste text from clipboard")
    #     async def paste_from_clipboard(browser: BrowserContext):
    #         text = pyperclip.paste()
    #         # send text to browser
    #         page = await browser.get_current_page()
    #         await page.keyboard.type(text)

    #         return ActionResult(extracted_content=text)

    #     @self.registry.action("Smart scroll up when stuck")
    #     async def smart_scroll_up(browser: BrowserContext):
    #         page = await browser.get_current_page()
    #         # Scroll to top of page
    #         await page.keyboard.press('Home')
    #         await asyncio.sleep(1)
    #         return ActionResult(extracted_content="Scrolled to top to recover from stuck state")
        
    #     @self.registry.action("Close popup or modal")
    #     async def close_popup(browser: BrowserContext):
    #         page = await browser.get_current_page()
    #         # Try common popup close selectors
    #         selectors = ['[aria-label="Close"]', '.close', '.modal-close', '[data-dismiss="modal"]', 'button:has-text("Close")', 'button:has-text("×")']
    #         for selector in selectors:
    #             try:
    #                 await page.click(selector, timeout=2000)
    #                 return ActionResult(extracted_content="Closed popup/modal")
    #             except:
    #                 continue
    #         # Try Escape key
    #         await page.keyboard.press('Escape')
    #         return ActionResult(extracted_content="Pressed Escape to close popup")
        
    #     @self.registry.action("Detect and handle stuck state")
    #     async def handle_stuck_state(browser: BrowserContext):
    #         page = await browser.get_current_page()
    #         current_url = page.url
    #         # Check if in footer or problematic area
    #         try:
    #             footer_element = await page.query_selector('footer, .footer, #footer')
    #             if footer_element:
    #                 await page.keyboard.press('Home')  # Go to top
    #                 return ActionResult(extracted_content="Detected footer area, scrolled to top")
    #         except:
    #             pass
    #         return ActionResult(extracted_content="State check completed")
    
    def _register_custom_actions(self):
        """Register all custom browser actions"""
        
        @self.registry.action("Copy text to clipboard")
        def copy_to_clipboard(text: str):
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)
    
        @self.registry.action("Paste text from clipboard")
        async def paste_from_clipboard(browser: BrowserContext):
            text = pyperclip.paste()
            page = await browser.get_current_page()
            await page.keyboard.type(text)
            return ActionResult(extracted_content=text)
        
        @self.registry.action("Close popup or dismiss overlay")
        async def close_popup(browser: BrowserContext):
            page = await browser.get_current_page()
            closed_something = False
            
            # Try specific popup close patterns
            close_patterns = [
                # Common close buttons
                'button[aria-label*="close" i]',
                'button[aria-label*="dismiss" i]',
                '[data-testid*="close"]',
                '[class*="close"]',
                '[class*="dismiss"]',
                # Modal specific
                '.modal-close',
                '.overlay-close',
                # X buttons
                'button:has-text("×")',
                'button:has-text("✕")',
                'span:has-text("×")',
                # Cookie/GDPR buttons
                'button:has-text("Accept")',
                'button:has-text("OK")',
                'button:has-text("Got it")',
                'button:has-text("Continue")',
                # Generic buttons that might close popups
                '[role="button"][aria-label*="close" i]'
            ]
            
            for pattern in close_patterns:
                try:
                    elements = await page.query_selector_all(pattern)
                    for element in elements:
                        if await element.is_visible():
                            await element.click(timeout=1000)
                            closed_something = True
                            await asyncio.sleep(0.5)
                            break
                    if closed_something:
                        break
                except:
                    continue
            
            # Try Escape key as backup
            if not closed_something:
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
            
            return ActionResult(extracted_content="Attempted to close popup/overlay")
        
        @self.registry.action("Scroll to top of page")
        async def scroll_to_top(browser: BrowserContext):
            page = await browser.get_current_page()
            await page.keyboard.press('Home')
            await asyncio.sleep(1)
            return ActionResult(extracted_content="Scrolled to top of page")
        
        @self.registry.action("Go to DuckDuckGo search")
        async def go_to_duckduckgo(browser: BrowserContext):
            page = await browser.get_current_page()
            await page.goto('https://duckduckgo.com')
            await asyncio.sleep(2)
            
            # Close any popups that might appear
            try:
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
            except:
                pass
            
            return ActionResult(extracted_content="Navigated to DuckDuckGo search engine")


        @self.registry.action("Auto popup killer - run on page load")
        async def auto_popup_killer(browser: BrowserContext):
            """Automatically detect and close popups, modals, and overlays"""
            page = await browser.get_current_page()
            popups_closed = 0
            
            try:
                # Step 1: Send ESC key immediately
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
                popups_closed += 1
                
                # Step 2: Common popup selectors
                popup_selectors = [
                    # Modal/overlay selectors
                    '.modal, .popup, .overlay, .lightbox',
                    '[class*="modal"], [class*="popup"], [class*="overlay"]',
                    '[id*="modal"], [id*="popup"], [id*="overlay"]',
                    
                    # Cookie banners
                    '.cookie-banner, .cookie-notice, .cookie-bar',
                    '[class*="cookie"], [class*="gdpr"], [class*="consent"]',
                    
                    # Newsletter/subscription popups
                    '[class*="newsletter"], [class*="subscribe"], [class*="signup"]',
                    
                    # Common close button patterns
                    '.close, .close-btn, .close-button',
                    '[aria-label*="close" i], [aria-label*="dismiss" i]',
                    'button:has-text("×"), button:has-text("✕"), button:has-text("Close")',
                    'button:has-text("No Thanks"), button:has-text("Maybe Later")',
                    'button:has-text("Accept"), button:has-text("OK")',
                    
                    # Notification bars
                    '.notification, .banner, .alert-bar, .promo-bar'
                ]
                
                # Step 3: Try to close visible popups
                for selector in popup_selectors:
                    try:
                        elements = await page.query_selector_all(selector)
                        for element in elements:
                            if await element.is_visible():
                                await element.click(timeout=1000)
                                popups_closed += 1
                                await asyncio.sleep(0.3)  # Brief pause between clicks
                                break  # Only close one per selector to avoid over-clicking
                    except:
                        continue  # Ignore errors, try next selector
                
                # Step 4: Handle specific popup types
                try:
                    # Cookie acceptance (common patterns)
                    cookie_buttons = [
                        'button:has-text("Accept All")',
                        'button:has-text("Accept Cookies")', 
                        'button:has-text("I Accept")',
                        'button:has-text("Continue")',
                        '[id*="accept"], [class*="accept"]'
                    ]
                    
                    for btn_selector in cookie_buttons:
                        try:
                            await page.click(btn_selector, timeout=1000)
                            popups_closed += 1
                            break
                        except:
                            continue
                except:
                    pass
                
                # Step 5: Final ESC attempt
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
                
                return ActionResult(extracted_content=f"Popup killer completed - {popups_closed} popups handled")
                
            except Exception as e:
                return ActionResult(extracted_content=f"Popup killer error: {str(e)}")
        
        @self.registry.action("Smart page preparation")
        async def smart_page_prep(browser: BrowserContext):
            """Prepare page for automation - kill popups and wait for load"""
            page = await browser.get_current_page()
            
            # Wait for page to be somewhat loaded
            try:
                await page.wait_for_load_state('domcontentloaded', timeout=5000)
            except:
                pass
            
            # Run popup killer
            popup_result = await auto_popup_killer(browser)
            
            # Additional wait for dynamic content
            await asyncio.sleep(1)
            
            return ActionResult(extracted_content=f"Page prepared for automation. {popup_result.extracted_content}")
        
        @self.registry.action("Search directly on DuckDuckGo with query")
        async def search_duckduckgo(query: str, browser: BrowserContext):
            """Search directly on DuckDuckGo using URL parameters - ENFORCED"""
            import urllib.parse
            encoded_query = urllib.parse.quote_plus(query)
            search_url = f"https://duckduckgo.com/?t=h_&q={encoded_query}"
            
            page = await browser.get_current_page()
            await page.goto(search_url)
            await asyncio.sleep(2)  # Wait for page to load
            
            return ActionResult(extracted_content=f"Searched DuckDuckGo for: {query}")
        
        # ADD new universal search that redirects to DuckDuckGo
        @self.registry.action("Search web (always uses DuckDuckGo)")
        async def search_web(query: str, browser: BrowserContext):
            """Universal search that always uses DuckDuckGo for privacy"""
            return await search_duckduckgo(query, browser)
        
        # OVERRIDE any Google search attempts
        @self.registry.action("Search Google (redirected to DuckDuckGo)")
        async def search_google(query: str, browser: BrowserContext):
            """Google search redirected to DuckDuckGo for privacy compliance"""
            logger.info(f"🔒 Google search blocked, using DuckDuckGo instead for: {query}")
            return await search_duckduckgo(query, browser)


        @self.registry.action("Detect and avoid captcha")
        async def captcha_avoidance(browser: BrowserContext):
            """Detect captcha and suggest alternative approaches"""
            page = await browser.get_current_page()
            
            # Common captcha indicators
            captcha_selectors = [
                '.captcha, .recaptcha, .hcaptcha',
                '[id*="captcha"], [class*="captcha"]',
                'iframe[src*="recaptcha"]',
                'iframe[src*="hcaptcha"]',
                '[class*="cf-challenge"]',  # Cloudflare
                '.g-recaptcha',
                '#recaptcha',
                'input[name*="captcha"]'
            ]
            
            captcha_detected = False
            captcha_type = "unknown"
            
            try:
                for selector in captcha_selectors:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        if await element.is_visible():
                            captcha_detected = True
                            if 'recaptcha' in selector:
                                captcha_type = "recaptcha"
                            elif 'hcaptcha' in selector:
                                captcha_type = "hcaptcha"
                            elif 'cloudflare' in selector or 'cf-challenge' in selector:
                                captcha_type = "cloudflare"
                            break
                    if captcha_detected:
                        break
            
            except Exception as e:
                return ActionResult(extracted_content=f"Captcha detection error: {str(e)}")
            
            if captcha_detected:
                logger.warning(f"🤖 Captcha detected: {captcha_type}")
                
                # Avoidance strategies
                avoidance_strategies = [
                    "Switch to mobile user agent",
                    "Try alternative website for same task", 
                    "Use different browser session",
                    "Wait and retry later",
                    "Inform user and request manual intervention"
                ]
                
                return ActionResult(
                    extracted_content=f"Captcha detected ({captcha_type}). Suggested alternatives: {', '.join(avoidance_strategies)}"
                )
            else:
                return ActionResult(extracted_content="No captcha detected - proceeding normally")
        
        @self.registry.action("Switch to mobile user agent")
        async def switch_mobile_agent(browser: BrowserContext):
            """Switch to mobile user agent to potentially avoid captchas"""
            page = await browser.get_current_page()
            
            mobile_agents = [
                "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1",
                "Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox/88.0"
            ]
            
            try:
                await page.set_extra_http_headers({
                    'User-Agent': mobile_agents[0]
                })
                await page.reload()
                await asyncio.sleep(2)
                
                return ActionResult(extracted_content="Switched to mobile user agent and reloaded page")
                
            except Exception as e:
                return ActionResult(extracted_content=f"Mobile agent switch failed: {str(e)}")


