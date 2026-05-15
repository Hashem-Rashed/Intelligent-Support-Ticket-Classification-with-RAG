#!/usr/bin/env python3
"""
Advanced synthetic ticket generator for 5 categories.
Produces 2M+ tickets with rich linguistic variation.

Usage:
    python generate_tickets.py --output data/raw/tickets.csv --num 2000000 --balanced
"""

import argparse
import csv
import random
import string
import itertools
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ------------------------------------------------------------
# Configuration – only 5 categories
# ------------------------------------------------------------
TARGET_CATEGORIES = ["Account", "Billing", "Fraud", "Technical", "General Inquiry"]

# Default realistic weights (based on real support data)
DEFAULT_WEIGHTS = {
    "Account": 0.22,
    "Billing": 0.22,
    "Fraud": 0.18,
    "Technical": 0.20,
    "General Inquiry": 0.18,
}
BALANCED_WEIGHTS = {cat: 0.2 for cat in TARGET_CATEGORIES}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def random_string(length=8, digits_only=False):
    if digits_only:
        return ''.join(random.choices(string.digits, k=length))
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def random_name():
    first = random.choice(["James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda",
                           "William","Elizabeth","David","Susan","Joseph","Jessica","Thomas","Sarah",
                           "Daniel","Lisa","Paul","Karen","Mark","Donna","Steven","Nancy","Andrew","Betty"])
    last = random.choice(["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
                          "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
                          "Thomas","Taylor","Moore","Jackson","Martin","Lee","White","Harris","Clark"])
    return f"{first} {last}"

def random_email(name):
    domains = ["example.com", "customer.org", "mail.net", "support.com", "user.me", "web.com"]
    local = name.lower().replace(" ", ".") + str(random.randint(1, 9999))
    return f"{local}@{random.choice(domains)}"

def random_date(start_year=2023, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

def random_phone():
    return f"+1 {random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"

def random_error_code():
    return random.choice(["500", "502", "503", "404", "401", "403", "429", "E001", "E002", "TIMEOUT"])

def random_amount():
    return round(random.uniform(4.99, 299.99), 2)

def random_product():
    return random.choice(["Pro Plan", "Basic", "Enterprise", "Premium", "Lite", "Mobile App", "Desktop App", "Web Dashboard"])

def random_device():
    return random.choice(["Windows 11", "macOS Ventura", "iPhone 14", "Android 13", "iPad", "Linux Ubuntu", "Chrome OS"])

def random_browser():
    return random.choice(["Chrome", "Firefox", "Safari", "Edge", "Opera"])

def random_action():
    return random.choice([
        "Please help me resolve this.",
        "I need assistance urgently.",
        "Can someone look into this?",
        "This is affecting my work.",
        "Please advise what to do next.",
        "I would appreciate a quick response.",
        "This is frustrating, please help.",
        "Thank you for your support."
    ])

def random_greeting():
    return random.choice([
        "Hello,", "Hi,", "Dear Support,", "Hey there,", "Good morning,", "Greetings,", ""
    ])

def random_closing():
    return random.choice([
        "Thanks,", "Best regards,", "Sincerely,", "Appreciate your help,", "Thank you,", "Regards,"
    ])

def maybe_typo(text, prob=0.05):
    """Introduce random typos to simulate real user input."""
    if random.random() > prob:
        return text
    # Simple typo: swap two adjacent characters
    chars = list(text)
    if len(chars) > 3:
        idx = random.randint(0, len(chars)-2)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    return ''.join(chars)

# ------------------------------------------------------------
# Template system with dynamic fragments
# ------------------------------------------------------------
class AdvancedTicketGenerator:
    def __init__(self):
        self.templates = self._build_templates()
        self.fragments = self._build_fragments()
        self._seen_texts = set()

    def _build_fragments(self):
        """Reusable sentence parts for dynamic composition."""
        return {
            "problem_statement": [
                "I'm having trouble with {feature}.",
                "There is an issue with {feature}.",
                "{feature} is not working as expected.",
                "I encountered a problem while using {feature}.",
                "Something went wrong with {feature}.",
            ],
            "specifics": [
                "Specifically, {detail}.",
                "The exact issue is: {detail}.",
                "More precisely, {detail}.",
                "To be clear, {detail}.",
            ],
            "request": [
                "Could you please fix this?",
                "Please resolve this as soon as possible.",
                "I would appreciate your help on this.",
                "Can you investigate and get back to me?",
                "Please advise what I should do next.",
            ]
        }

    def _build_templates(self):
        """Rich templates for each category (15-20 per category)."""
        templates = defaultdict(list)

        # ----- Account (18 templates) -----
        templates["Account"].extend([
            "I can't log into my account. {action} I've tried resetting my password but the email never arrives. My username is {username}.",
            "Every time I try to access my account, I get a 'suspicious activity detected' message. {action} I'm sure it's my own login – please unlock my account. User ID: {user_id}.",
            "Two-factor authentication is not working. I enter the code from my authenticator app but it says 'invalid code'. {action} I need access urgently.",
            "My account has been locked after several failed login attempts. {action} I was trying to reset my password because I forgot it, but now I'm completely locked out.",
            "I changed my phone number and can't receive 2FA codes. {action} Please update my account to use my new number: {phone}.",
            "The 'forgot password' link does nothing – no email arrives. {action} I've checked spam folder. Please help me regain access.",
            "I'm getting 'account disabled' message. {action} I haven't violated any terms. Can you re-enable it? My email is {email}.",
            "I tried to sign up but it says email already exists – that's my email but I never created an account. {action}",
            "I'm locked out because my company changed my email domain. {action} Old: {email}, new: {email}. Please migrate my data.",
            "Every time I log in, I'm redirected to a blank page. {action} I've cleared cookies. Works on incognito? No.",
            "I keep getting 'session expired' every 2 minutes. {action} This makes the platform unusable.",
            "My account was deactivated for inactivity, but I need to access old invoices. {action} Please reactivate temporarily.",
            "I've forgotten my security questions. {action} Is there another way to verify my identity?",
            "I'm trying to merge two accounts but the tool fails. {action} One is personal, one is work. Both have valid subscriptions.",
            "The account recovery process asks for a code sent to a phone number I no longer have. {action}",
            "I receive login alerts from unknown locations – probably VPN, but I want to whitelist my IP. {action}",
            "SSO login stopped working after our IdP upgrade. {action} Error: {error_code}. Need to re-establish trust.",
            "My account shows someone else's name after a glitch. {action} It's displaying '{random_name}' but I'm {name}. Please correct."
        ])

        # ----- Billing (18 templates) -----
        templates["Billing"].extend([
            "I was charged twice for my subscription this month. {action} The duplicate charge appears on my {date} statement. Please refund the extra amount.",
            "My invoice shows an incorrect amount. I'm on the basic plan ($9.99/month) but you charged me $29.99. {action} Please correct my billing plan.",
            "I cancelled my subscription on {date} but you still charged me this month. {action} Kindly refund the charge and confirm cancellation.",
            "The payment method on file is expired. I tried updating my credit card but the system says 'invalid card'. {action} I don't want service interruption.",
            "I need a copy of my invoice for tax purposes. {action} My account email is {email}.",
            "Why was I charged an extra ${amount}? This never happened before. {action} Please check my billing history.",
            "I want to downgrade my plan but the system keeps showing an error. {action} Please process the downgrade manually.",
            "I upgraded my plan mid-month and was charged full price for both. {action} Pro-rated amount expected.",
            "There's a pending transaction that hasn't cleared in 10 days. {action} It's blocking my account upgrade.",
            "My discount coupon wasn't applied. Code: {random_string(8)}. {action} Please adjust the invoice.",
            "I'm being charged for a free trial I never converted. {action} I canceled before the trial ended.",
            "The invoice currency is wrong (USD instead of EUR). {action} My billing address is in Germany.",
            "I received a receipt for a $0 invoice – seems like a bug. {action} Please confirm no action needed.",
            "Auto-renew was supposed to be off, but it renewed anyway. {action} I want a refund and cancellation.",
            "The VAT amount on my invoice is incorrect. {action} My VAT ID is {random_string(10)}. Please reissue.",
            "I paid via bank transfer a week ago but my account still shows unpaid. {action} Reference: {random_string(12)}.",
            "The subscription started on the wrong date. {action} It should begin on {date} not earlier.",
            "I see multiple pending charges from your company – I only have one plan. {action} Please investigate."
        ])

        # ----- Fraud (18 templates) -----
        templates["Fraud"].extend([
            "Someone made several unauthorized purchases using my credit card. {action} I see charges for {amount} on {date}. I did NOT authorize these.",
            "My account was hacked – the hacker changed my email and password. {action} I can't log in anymore. Please lock everything immediately!",
            "I received an email saying my account was accessed from a new device in {country}. I didn't do that. {action} Please secure my account.",
            "There are multiple fraudulent refund requests submitted from my account. {action} I never requested any refunds. This is identity theft.",
            "I got a suspicious payment confirmation for a transaction I never made. {action} The payment ID is {trans_id}. Please investigate.",
            "Someone is trying to reset my password repeatedly. {action} I got 10 password reset emails in one hour. Please lock my account temporarily.",
            "A charge from your company appeared on my card but I never signed up for any service. {action} This is fraud. Reverse the charge immediately.",
            "I received a phishing email that looked exactly like your official communication. {action} It asked for my password. Please warn others.",
            "My account was used to post spam messages. {action} I have never posted anything. Please restore my reputation.",
            "I see login attempts from {country} every few hours. {action} Please enable geographic restrictions on my account.",
            "The hacker changed my recovery email to {email}. {action} That's not mine. Please revert to {email}.",
            "Someone purchased a gift card using my saved payment method. {action} I have never bought gift cards.",
            "I got a call from someone claiming to be your support asking for my credit card. {action} I hung up. Is that normal?",
            "My account shows a new API key that I didn't create. {action} Please revoke it and secure my account.",
            "I noticed a strange device listed in 'trusted devices': '{device}'. {action} I don't own that device.",
            "The fraud alert system is too sensitive – my legitimate purchases are being blocked. {action} Please adjust thresholds.",
            "I received a notification that my personal data was exposed in a breach. {action} What information was leaked?",
            "A transaction was flagged as fraud but it's actually mine (I was traveling). {action} How can I verify my identity to unblock?"
        ])

        # ----- Technical (20 templates) -----
        templates["Technical"].extend([
            "The app crashes every time I try to open the dashboard. {action} I'm on version {version}, using {device}. Error: {error_code}.",
            "The website is extremely slow and images don't load. {action} I've cleared cache and tried different browsers, still the same.",
            "I get a 500 internal server error when I submit the form. {action} I've tried on {browser} and Firefox. Please fix the backend.",
            "The mobile app won't sync my data. {action} I have the latest version. It shows 'network error' even though my internet works.",
            "Dark mode doesn't work properly – text is unreadable. {action} Screenshot attached. Please fix this UI bug.",
            "The search function returns no results even for obvious queries. {action} Example: searching for 'invoice' shows nothing even though I have invoices.",
            "After the latest update, the app freezes for 10 seconds on startup. {action} This happens every time. Device: {device}.",
            "I can't upload files larger than 2MB – the limit says 10MB. {action} Error: '413 Payload Too Large'.",
            "Push notifications are delayed by hours. {action} I receive them long after the event.",
            "The API endpoint /v2/users returns a 403 even with valid token. {action} It worked yesterday.",
            "When I export data to CSV, special characters are corrupted. {action} Example: 'é' becomes 'Ã©'.",
            "The calendar widget shows the wrong timezone (UTC instead of local). {action} My profile timezone is set correctly.",
            "Video playback stutters on {browser} but works on Chrome. {action} Please investigate compatibility.",
            "The 'remember me' checkbox doesn't work – I have to log in every hour. {action}",
            "I'm getting a 'WebSocket connection failed' error in the console. {action} Real-time updates are broken.",
            "The mobile app drains battery excessively (30% per hour). {action} Something is wrong with background processes.",
            "I can't delete my project – the delete button is greyed out. {action} I'm the owner.",
            "The chart on the dashboard shows incorrect data after midnight. {action} It seems to be a UTC conversion bug.",
            "When I use screen reader, the buttons are not labeled. {action} Accessibility issue.",
            "The PDF export cuts off the right side of the table. {action} Using landscape mode doesn't help."
        ])

        # ----- General Inquiry (16 templates) -----
        templates["General Inquiry"].extend([
            "Hello, I have a quick question about your pricing plans. {action} Could you send me a comparison of the premium vs enterprise?",
            "What are your business hours for customer support? {action} I'm in a different time zone and want to know when to call.",
            "Do you offer discounts for non‑profits? {action} Our organization is small and we'd love to use your service if it's affordable.",
            "Is there a way to export my data as a CSV file? {action} I need to run some analysis externally.",
            "Can I upgrade my plan mid‑billing cycle? {action} I need more features now, but I don't want to waste the remaining days.",
            "Where can I find documentation for your API? {action} I'm a developer and need integration details.",
            "Do you have a mobile app for iOS? {action} I couldn't find it in the App Store.",
            "How long does it take to get a response from support? {action} I'm on a trial and need help quickly.",
            "What's the difference between the Basic and Pro plans? {action} I can't find a feature matrix.",
            "Can I use your service for my team of 50 people? {action} Is there a volume discount?",
            "Is there a way to test your service without providing credit card? {action} I want to evaluate first.",
            "Do you integrate with Slack or Teams? {action} We need notifications in our chat.",
            "What security certifications do you have? {action} Our compliance team requires SOC2.",
            "How can I delete my account permanently? {action} I've decided to move to another service.",
            "Is there a community forum where users discuss tips? {action} I'd like to learn from others.",
            "Can I schedule a demo with a sales representative? {action} Please let me know available slots."
        ])

        return templates

    def _generate_rich_description(self, category):
        """Dynamically compose description from fragments and templates."""
        template = random.choice(self.templates[category])
        # Add optional greeting and closing
        greeting = random_greeting()
        closing = random_closing()
        # Replace placeholders
        placeholders = {
            "{action}": random_action(),
            "{username}": random_string(8),
            "{user_id}": random_string(6).upper(),
            "{email}": random_email(random_name()),
            "{date}": random_date().strftime("%B %d, %Y"),
            "{amount}": f"${random_amount():.2f}",
            "{country}": random.choice(["Russia", "China", "Brazil", "Nigeria", "Ukraine", "India", "Vietnam"]),
            "{trans_id}": "TXN" + random_string(10).upper(),
            "{version}": f"{random.randint(1,4)}.{random.randint(0,10)}.{random.randint(0,20)}",
            "{error_code}": random_error_code(),
            "{phone}": random_phone(),
            "{device}": random_device(),
            "{browser}": random_browser(),
            "{feature}": random.choice(["login", "dashboard", "reporting", "export", "notification", "sync", "profile"]),
            "{detail}": random.choice(["the button does nothing", "the page is blank", "it says 'access denied'", "the data is outdated"]),
            "{name}": random_name(),
            "{random_string(8)}": random_string(8),
            "{random_string(10)}": random_string(10),
            "{random_string(12)}": random_string(12),
        }
        desc = template
        for key, value in placeholders.items():
            desc = desc.replace(key, value)
        # Add variation: sometimes insert an extra sentence
        if random.random() < 0.3:
            extra = random.choice([
                " I've attached a screenshot for reference.",
                " Let me know if you need more details.",
                " I'm happy to provide logs if necessary.",
                " This has been happening for a week.",
                " I tried reinstalling but no change."
            ])
            desc += extra
        # Assemble final with greeting/closing (10% chance to skip greeting)
        if greeting and random.random() > 0.3:
            desc = f"{greeting} {desc}"
        if closing and random.random() > 0.5:
            desc = f"{desc} {closing}"
        # Optionally add a typo
        desc = maybe_typo(desc, prob=0.02)
        return desc.strip()

    def generate_ticket(self, category):
        """Generate a unique ticket description."""
        for _ in range(10):
            desc = self._generate_rich_description(category)
            if desc not in self._seen_texts:
                self._seen_texts.add(desc)
                return desc
        # Fallback: add random suffix
        desc = desc + " " + random_string(5)
        self._seen_texts.add(desc)
        return desc


# ------------------------------------------------------------
# Main generation function
# ------------------------------------------------------------
def generate_dataset(output_path: Path, num_rows: int, balanced: bool = False, batch_size: int = 10000):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights = BALANCED_WEIGHTS if balanced else DEFAULT_WEIGHTS

    # Build category list with exact counts
    categories = []
    total = 0
    for cat, weight in weights.items():
        count = int(num_rows * weight)
        categories.extend([cat] * count)
        total += count
    # Adjust remainder (due to rounding) – assign to General Inquiry
    remainder = num_rows - total
    if remainder > 0:
        categories.extend(["General Inquiry"] * remainder)
    random.shuffle(categories)

    generator = AdvancedTicketGenerator()
    row_count = 0
    batch_rows = []

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Ticket_ID", "Customer_Name", "Customer_Email", "Ticket_Subject",
            "Ticket_Description", "Issue_Category", "Priority_Level",
            "Ticket_Channel", "Submission_Date", "Resolution_Time_Hours",
            "Assigned_Agent", "Satisfaction_Score"
        ])

        for i, category in enumerate(categories):
            desc = generator.generate_ticket(category)
            row = [
                f"TICKET-{i+1:08d}",
                None, None, None,
                desc, category,
                None, None, None, None, None, None
            ]
            batch_rows.append(row)
            row_count += 1

            if row_count % batch_size == 0:
                writer.writerows(batch_rows)
                batch_rows = []
                print(f"Generated {row_count:,} / {num_rows:,} tickets", flush=True)

        if batch_rows:
            writer.writerows(batch_rows)

    print(f"\n✅ Dataset saved to {output_path}")
    print(f"Total rows: {row_count:,}")
    print(f"Unique descriptions: {len(generator._seen_texts)} (all unique)")
    print("\nCategory distribution (actual):")
    from collections import Counter
    cnt = Counter(categories)
    for cat in TARGET_CATEGORIES:
        print(f"  {cat}: {cnt[cat]:,} ({cnt[cat]/num_rows*100:.1f}%)")


# ------------------------------------------------------------
# Command line
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced synthetic ticket generator (5 categories, 2M+ rows)")
    parser.add_argument("--output", type=str, default="data/raw/tickets.csv",
                        help="Output CSV file path")
    parser.add_argument("--num", type=int, default=2000000,
                        help="Number of tickets to generate")
    parser.add_argument("--balanced", action="store_true",
                        help="Generate equal number of samples per category (20% each)")
    parser.add_argument("--batch", type=int, default=10000,
                        help="Batch size for writing")
    args = parser.parse_args()

    output_file = Path(args.output)
    generate_dataset(output_file, args.num, balanced=args.balanced, batch_size=args.batch)