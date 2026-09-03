# RecoverAI: The Complete Beginner-Friendly Study Guide
### *A plain-English, code-grounded guide to understanding how this project works from scratch*

---

## 1. The Problem (Plain English)

Imagine you run an online clothing store. Every day, hundreds of customers pick items, enter their details, and hit **"Pay Now"**. But roughly 1 out of every 10 payments fails. 

Sometimes it's just a tiny hiccup: the customer's phone Wi-Fi blinked for two seconds, or their bank app was temporarily overloaded. Other times, it's a permanent dead end: their card expired six months ago, their bank account is empty, or a hacker is testing stolen card numbers.

Today, most online businesses handle this in one of two flawed ways:
1. **They do nothing:** The customer gets frustrated, closes their browser tab, and walks away. You lose the sale, and you might lose that customer forever.
2. **They use blind, automated retries:** A computer script blindly attempts to charge the failed card again every 4 hours. 
   - If the failure was just a 2-second Wi-Fi blink, retrying might work.
   - But if the card was expired, retrying it 5 times fails all 5 times. Even worse, payment gateways (like Razorpay, Visa, or Mastercard) charge the store owner a fee for every attempt, and banks can penalize stores that repeatedly submit dead cards.
   - And if the transaction was attempted by a fraudster using stolen credentials, blindly retrying helps the criminal, leaving the store owner on the hook for massive chargeback fines.

**RecoverAI solves this problem by acting like an experienced, cautious financial analyst who watches every failed payment in real time.** The moment a payment drops, it asks: *“Is this customer actually likely to pay if we reach out? Does the money we might recover justify the small fee of sending a payment link? And is it 100% safe and fraud-free?”* If yes, it steps in. If no, it steps aside.

---

## 2. The One-Paragraph Elevator Pitch

> *"RecoverAI is a smart, safety-first revenue recovery system for online merchants. When a customer's payment fails at checkout, instead of blindly retrying and burning gateway fees, RecoverAI uses machine learning to calculate the exact probability that the payment can be saved. It does the math to make sure reaching out is actually profitable, uses an AI model to diagnose the technical error, and subjects every decision to a strict, hard-coded safety policy. That policy halts high-value orders above ₹8,500 for human review and permanently blocks fraud. Finally, it only counts money as recovered when a cryptographically verified webhook from Razorpay proves the customer's payment has settled in the bank."*

---

## 3. The End-to-End Flow (Plain English)

Here is the journey of a single failed payment, from the second it fails to the second money is saved:

1. **A payment fails at checkout:** A customer tries to buy a ₹2,500 jacket, but their payment fails due to a network glitch or a bank error.
2. **The system gathers the story (Context):** The system collects 31 pieces of factual background data about the customer and transaction—such as how long the customer has had an account, how many successful orders they've made in the past, their previous failure streak, the time of day, and their IP risk score.
3. **The ML model predicts the chance of success:** A trained machine learning model looks at those 31 pieces of data and calculates a single percentage: for example, *"There is a 98.5% chance this payment can be recovered if we send a payment link."*
4. **The system calculates Expected Value (The Math Check):** The system checks if intervening makes financial sense. If generating a payment link costs ₹12, but we have a 98.5% chance of saving a ₹2,500 order, our expected return is over ₹2,400. That is a clear mathematical win.
5. **The AI diagnoses the technical error:** An AI language model reads the bank's error message, explains what went wrong in plain English (e.g., *"Temporary bank gateway outage"*), and recommends the best recovery tactic (e.g., *"Send a fresh Razorpay payment link"*).
6. **The Policy Guard checks the safety rules:** Before anything is allowed to execute, a strict rules engine checks non-negotiable business rules:
   - *Is the fraud score high?* $\to$ **REFUSE** immediately. No money moves, no links are sent.
   - *Is the order big (₹8,500 or more)?* $\to$ **ESCALATE** to human review. The automated pipeline pauses.
   - *Is the chance of recovery below 35%?* $\to$ **REFUSE**. Don't waste money chasing lost causes.
   - *Did it pass every single safety check?* $\to$ **ACT**. Proceed to send the link.
7. **Safe execution with double-click protection:** If cleared to ACT, the system contacts Razorpay's API to generate a fresh payment link. An automatic database lock guarantees that even if the customer or server double-clicks, only one link is ever generated.
8. **Waiting in open loop:** The payment is marked as `PROCESSING`. The store owner does **not** count the money as saved yet, because the customer hasn't actually paid.
9. **Customer pays via the link:** The customer receives the link on their phone, opens it, enters their UPI or card, and completes the payment.
10. **Cryptographic proof closes the loop:** Razorpay's servers send an official notification (a webhook) to our server with a cryptographic digital signature proving it is genuinely from Razorpay. Only after verifying this signature does our system officially mark the order as **`RECOVERED`**.

---

## 4. Key Concepts Glossary

| Concept | Plain English Definition & Everyday Analogy | Why It Matters Here |
| :--- | :--- | :--- |
| **Recovery Probability ($P$)** | A number between `0.0` (0%) and `1.0` (100%) that estimates how likely a failed payment is to succeed if we try to recover it. <br>*Analogy: A weather forecast saying there is an 85% chance of rain.* | It stops us from guessing blindly. We treat an 85% likely bank hiccup very differently from a 3% likely dead account. |
| **Expected Value (EV)** | The average money you gain from taking an action after subtracting what it costs to take that action: `(Chance of Success × Order Amount) - Cost to Try`. <br>*Analogy: Spending ₹10 on a flashlight battery to find a dropped ₹1,000 note in the dark is a great deal (+₹790 EV). Spending ₹10 on a battery to look for a 10-rupee coin is a waste of money.* | It tells us if an intervention is profitable. It prevents spending ₹12 chasing a ₹50 order that will probably fail anyway. |
| **Policy Engine** | A strict list of hard-coded `if/then` business rules written in Python that can never be bypassed or argued with by AI. <br>*Analogy: A nightclub bouncer checking physical IDs at the door. It doesn't matter what your friend says—if you don't have an ID, you don't get in.* | Machine learning and AI models only make suggestions. The policy engine guarantees that fraud is permanently blocked and big orders are checked by humans. |
| **Idempotency** | A fancy engineering word that means: *doing something five times produces the exact same result as doing it once.* <br>*Analogy: An elevator button. If you press the "Floor 4" button once, the elevator goes to the 4th floor. If you mash it 10 times rapidly, it still only goes to the 4th floor once.* | It prevents double-charges. If an admin double-clicks "Approve", or a network glitch sends two identical requests, only one payment link is created. |
| **Webhook** | An automated notification sent from one server to another the second an event happens. <br>*Analogy: Instead of you calling the pizza shop every 2 minutes asking "Is my pizza here yet?", the delivery driver rings your doorbell when it arrives.* | Instead of our server repeatedly asking Razorpay every 10 seconds "Did the customer pay?", Razorpay knocks on our server's door the exact millisecond payment clears. |
| **HMAC Signature** | A secret digital wax seal attached to a webhook message that proves it genuinely came from Razorpay and wasn't faked by a scammer. <br>*Analogy: A wax seal stamped with the King's private ring on a royal letter. If the wax is broken or the stamp is wrong, you know it's a fake.* | Anyone on the internet can send an HTTP request to our server claiming "Order #123 was paid!" The HMAC signature proves it came from Razorpay and rejects impostors. |
| **LangGraph / StateGraph** | A Python coding library used to build step-by-step AI workflows as an orderly flowchart (called a "graph"). <br>*Analogy: An assembly line in a car factory. Station 1 adds the wheels, Station 2 inspects the engine, Station 3 paints the car. Each station passes the car to the next.* | It organizes the whole recovery workflow into clean, isolated steps: Node 1 loads data, Node 2 runs ML, Node 3 runs AI diagnosis, Node 4 checks policy rules. |
| **PII Redaction** | Automatically finding and masking Personally Identifiable Information (customer names, phone numbers, email addresses, credit card numbers). <br>*Analogy: A government document with black marker blacking out secret names before sharing it with the public.* | Keeps merchants compliant with privacy laws (PCI-DSS) by guaranteeing customer credit card numbers and phone numbers are never sent to cloud AI providers like OpenAI. |
| **Human-in-the-Loop (HITL)** | Intentionally pausing automated code and asking a human to review and click "Approve" before taking action. <br>*Analogy: A bank's automatic fraud alert that freezes a huge ₹10,00,000 transfer until a human bank manager calls you to confirm.* | Keeps store owners in full control of large financial amounts ($\ge ₹8,500$) where an automated AI mistake would be too costly. |

---

## 5. The ML Layer, Explained With Real Numbers

### What the Model Predicts
The machine learning model is a calibrated **Logistic Regression** pipeline (saved in `ml/models/recovery_model.joblib`). It takes 31 factual features about the transaction and outputs a single decimal number between `0.0000` and `1.0000`: the probability of recovery.

### Real Input Row Example
Here is an exact row of transaction features from the project's dataset:

```python
{
    "amount": 2500.0,                   # Order amount is ₹2,500
    "amount_vs_customer_average": 1.05, # Close to their usual order size
    "hour": 14,                         # 2:00 PM in the afternoon
    "day_of_week": 2,                   # Tuesday
    "customer_previous_transactions": 12, # Customer has ordered 12 times before
    "customer_successful_transactions": 11,# 11 of those 12 succeeded (reliable buyer)
    "customer_historical_success_rate": 0.916, # 91.6% past success rate
    "consecutive_failure_streak": 0,    # This is their first failure today
    "ip_risk_score": 0.04,              # Very clean IP (only 4% risk score)
    "velocity_score": 0.12,             # Normal speed (not a bot spamming orders)
    "payment_method": "card",           # Tried paying with a credit card
    "failure_reason": "network_timeout",# Bank took too long to respond
    "failure_category": "transient"     # A temporary blip, not a broken card
}
```

When this dictionary is passed to our model (`ml/predict.py`), the model outputs:
**`0.9855`** $\to$ **a 98.55% probability that this payment can be recovered!**

### Expected Value Math Walked by Hand
Every action we take has a real cost. In `evaluation/business_metrics.py`, the cost structure is defined:
* **`retry`** (automatic background gateway retry): **₹5.00**
* **`reminder`** (sending an automated WhatsApp/SMS alert): **₹2.00**
* **`payment_link`** (generating and emailing a dedicated payment link): **₹12.00**
* **`no_action`** (walking away): **₹0.00**

Here is the formula:
$$\text{Expected Value} = (\text{Probability} \times \text{Order Amount}) - \text{Action Cost}$$

#### Example 1: The ₹2,500 Good Order (Scenario A)
- **Order Amount:** ₹2,500.00
- **Probability ($P$):** 0.9855 (98.55%)
- **Assigned Action:** Payment Link (Cost = ₹12.00)

1. Multiply the chance by the money:
   $$0.9855 \times ₹2,500.00 = ₹2,463.75 \text{ (Gross Expected Value)}$$
2. Subtract what it costs to send the link:
   $$₹2,463.75 - ₹12.00 = \mathbf{+₹2,451.75} \text{ (Net Expected Value)}$$

**Conclusion:** We expect to make **+₹2,451.75** on average by spending ₹12. It is a massive financial win to step in!

#### Example 2: The ₹150 Lost Cause
- **Order Amount:** ₹150.00
- **Probability ($P$):** 0.05 (Only a 5% chance—e.g., bad card, brand new user)
- **Assigned Action:** Payment Link (Cost = ₹12.00)

1. Multiply the chance by the money:
   $$0.05 \times ₹150.00 = ₹7.50 \text{ (Gross Expected Value)}$$
2. Subtract what it costs to send the link:
   $$₹7.50 - ₹12.00 = \mathbf{-₹4.50} \text{ (Net Expected Value)}$$

**Conclusion:** The expected value is **negative** (-₹4.50). If you chase 100 orders like this, you will spend ₹1,200 in gateway fees but only recover ₹750! The smartest financial move is to do nothing (`no_action`).

### The Real Code
In `agent/nodes/prediction.py`, lines 26–30 execute this calculation directly:

```python
res = predict_recovery_probability(ml_input)
prob = float(res["recovery_probability"])

amount = float(state.get("amount", 0.0))
expected_val = round(amount * prob, 2)
```

---

## 6. The Policy Engine, Explained With Real Thresholds

### Why Raw ML Probability Alone Isn't Safe
A machine learning model is just a statistical calculator. It has no common sense:
- What if an order is for **₹95,000**? Even if the model says there is an 80% chance of recovery, allowing an autonomous AI script to charge ₹95,000 without a human merchant double-checking is a dangerous financial liability.
- What if the model sees an 85% success chance, but the transaction was submitted from a stolen IP address with an IP risk score of 0.92? The ML model might miss the fraud risk, but the store owner would get hit with a chargeback penalty.

That is why RecoverAI has a **Deterministic Policy Guard** (`agent/nodes/policy.py`). It is a hard wall of code that runs after the ML model and overrides everything if a safety rule is violated.

### The Three Policy Outcomes

```text
                               FAILED PAYMENT
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
     🔴 REFUSE                  🟡 ESCALATE                   🟢 ACT
  (Action Blocked)        (Human Review Required)       (Cleared for Execution)
  - IP Risk > 0.70        - Amount >= ₹8,500            - Passed all safety rules
  - Expired / Bad Card    - Probability 0.32 to 0.38    - Probability >= 0.35
  - Streak >= 4 failures  - High velocity warning       - Amount < ₹8,500
  - Probability < 0.35
```

#### 1. 🔴 REFUSE (Permanent Block)
The system completely stops. No payment links are created, no API calls are made, and zero fees are spent.
* **Rule A (Fraud / Suspicious IP):** If `ip_risk_score > 0.70` or `failure_reason == "suspected_risk"`.
* **Rule B (Permanent Card Defect):** If the card is physically broken or expired (`card_expired`, `invalid_card`).
* **Rule C (Streak Limit):** If the card has failed 4 times in a row (`consecutive_failure_streak >= 4`).
* **Rule D (Unprofitable):** If the recovery probability is less than 35% (`recovery_probability < 0.35`).
* **Worked Example:** A payment fails with amount = ₹7,500 and `ip_risk_score = 0.92`.
  - **Verdict:** **`REFUSE`** (Reason: `HARD_FRAUD_BLOCK`). Prohibited from human override.

#### 2. 🟡 ESCALATE (Paused for Merchant Approval)
Automated execution pauses. The transaction is placed into the **Merchant Approval Queue** on the dashboard.
* **Rule A (High-Value Order):** If `amount >= 8500.00` (₹8,500 is the 95th percentile cutoff of all orders).
* **Rule B (Uncertainty Zone):** If the model isn't sure, falling right around the threshold: `0.32 <= recovery_probability <= 0.38`.
* **Worked Example:** A good customer buys an expensive TV for ₹14,500. The recovery probability is 76.7%.
  - **Verdict:** **`ESCALATE`** (Reason: `HIGH_VALUE_TRANSACTION`). It is paused because ₹14,500 $\ge$ ₹8,500.

#### 3. 🟢 ACT (Safe for Autonomous Action)
The transaction is safe, profitable, and under the high-value limit.
* **Rules:** Probability $\ge 0.35$, amount $< ₹8,500$, IP risk $\le 0.70$, failure streak $< 4$.
* **Worked Example:** Amount = ₹2,500, network timeout, probability = 0.9855, IP risk = 0.04.
  - **Verdict:** **`ACT`** (Reason: `STANDARD_POLICY_APPROVAL`). Proceeds to send a payment link.

### The Real Code
In `agent/nodes/policy.py`, lines 21–26 and 100–125 define these exact numbers:

```python
HIGH_VALUE_TRANSACTION_THRESHOLD: float = 8500.00
UNCERTAINTY_BAND_LOW: float = 0.32
UNCERTAINTY_BAND_HIGH: float = 0.38

# Stage 1: Hard Refusal Checks (Blocks fraud, expired cards, low probability)
if ip_risk > 0.70 or reason in ["suspected_risk"]:
    return "REFUSE", "HARD_FRAUD_BLOCK", "no_action"
if streak >= 4:
    return "REFUSE", "EXCESSIVE_FAILURE_STREAK", "no_action"
if pred_prob < 0.35:
    return "REFUSE", "BELOW_VIABILITY_THRESHOLD", "no_action"

# Stage 2: Escalation Checks (High-value or uncertain)
if amount >= HIGH_VALUE_TRANSACTION_THRESHOLD:
    return "ESCALATE", "HIGH_VALUE_TRANSACTION", rec_action
if UNCERTAINTY_BAND_LOW <= pred_prob <= UNCERTAINTY_BAND_HIGH:
    return "ESCALATE", "BOUNDARY_UNCERTAINTY_ESCALATE", rec_action
```

---

## 7. The LLM's Actual Role

### What the AI Does and Does NOT Do
There is a massive misconception in many "AI" projects that the LLM is given direct access to bank accounts and API keys. In fintech, that is reckless. 

* **What the LLM DOES:** It acts like a medical doctor reviewing an X-ray. It reads the raw error code from the bank, customer history, and payment channel. It writes a structured diagnosis explaining *why* the failure happened and what recovery strategy makes sense.
* **What the LLM NEVER DOES:** **The LLM has zero execution authority.** It does not hold Razorpay API keys. It cannot make HTTP requests to payment gateways. It cannot click "charge" or "send link". It only fills out a structured advisory form (`DiagnosisOutput`).

Even if the LLM completely hallucinates and says: *"This card is expired, but I recommend charging ₹100,000 right now!"*, the downstream deterministic policy engine reads the rule `card_expired` and instantly slaps a **`REFUSE`** on it. The LLM's advice is completely ignored.

### What Happens If the LLM Goes Down?
What if OpenAI or Groq has an outage, or our internet drops, or we hit an HTTP 429 rate limit? **RecoverAI does not crash.**

In `agent/services/llm_service.py`, lines 150–185 contain a built-in rule-based fallback heuristic:

```python
def _heuristic_diagnosis(context: Dict[str, Any]) -> DiagnosisOutput:
    """Fallback rule-based diagnosis when LLM is unavailable."""
    reason = str(context.get("failure_reason", "")).lower()
    if reason in ["network_timeout", "technical_error"]:
        return DiagnosisOutput(
            failure_category="transient",
            diagnosis="Temporary gateway disruption. Recommended for immediate retry.",
            severity="LOW"
        )
```

If the LLM is offline, the code catches the error, logs a warning, swaps in this rule-based diagnosis, and the pipeline continues running smoothly with zero downtime.

---

## 8. Safety Mechanisms, Explained With the Problem They Solve First

### 1. Idempotency (The Double-Payment Problem)
* **The Real-World Problem:** Imagine an admin is looking at a ₹14,500 order in the dashboard and accidentally double-clicks the "Approve" button. Or imagine the payment gateway's network glitched and sent two identical webhook notifications 10 milliseconds apart. Without protection, your server would make two API calls to Razorpay, creating two separate payment links or charging the customer twice.
* **How RecoverAI Fixes It:** In `payment/executor.py` and `database/models.py`, RecoverAI uses a two-tier lock:
  1. An in-memory thread lock: `_LOCAL_EXECUTION_LOCK = threading.Lock()`.
  2. A database table called `PaymentExecutionClaim` where the `idempotency_key` (the transaction ID) is set as the **Primary Key**.
  
  When an action begins, the code attempts to save a claim row to the database. The database allows the first insert. If a duplicate request arrives 5 milliseconds later, the database rejects it with an integrity error. The code catches this and simply returns the existing payment link without ever calling Razorpay a second time.

### 2. Webhook Signature Verification (The Fake-Webhook Problem)
* **The Real-World Problem:** Your server has an open doorway on the internet: an API endpoint at `/api/v1/webhooks/razorpay`. Anyone who finds that URL could use their laptop to send a fake HTTP POST request saying: `{"event": "payment_link.paid", "amount": 50000, "status": "paid"}`. If your code blindly believed incoming webhooks, a hacker could trick your system into marking expensive unpaid orders as "settled" and getting items shipped for free.
* **How RecoverAI Fixes It:** In `payment/webhook.py`, every single incoming webhook must pass an **HMAC-SHA256 signature check**:

```python
expected_sig = hmac.new(
    secret.encode("utf-8"),
    raw_body,
    hashlib.sha256
).hexdigest()

if not hmac.compare_digest(expected_sig, incoming_signature):
    raise HTTPException(status_code=400, detail="Invalid webhook signature")
```

Razorpay takes the message, mixes it with a private secret key only Razorpay and you know, and creates a unique cryptographic signature. Our server recalculates that exact math on the raw bytes. If a single letter was faked, or the secret key didn't match, the request is instantly rejected with HTTP 400.

### 3. Decoupling Link Creation from Settlement (The Phantom Revenue Problem)
* **The Real-World Problem:** Many amateur recovery tools send an email with a payment link to a customer and immediately celebrate on their dashboard: *"₹10,000 Recovered!"* In the real world, over 40% of customers look at the link, decide they don't want the item anymore, and never pay. Calling a link "recovered revenue" before the customer pays is financial fiction (phantom revenue).
* **How RecoverAI Fixes It:** In our database (`database/models.py`), creating a link only sets the status to **`PROCESSING`**. The status is **only** allowed to change to **`RECOVERED`** when Razorpay's cryptographically signed webhook arrives proving the customer's money has landed.

### 4. Defensive PII Redaction (The Data-Leak Problem)
* **The Real-World Problem:** You cannot send raw checkout data to an external AI service like OpenAI or Groq. If customer credit card numbers, phone numbers, or email addresses end up in cloud AI prompts, you violate banking privacy laws (PCI-DSS) and face massive regulatory fines.
* **How RecoverAI Fixes It:** In `agent/services/pii_redaction.py`, before any data is sent to the LLM, the function `redact_pii_from_context` creates an isolated deep copy of the transaction. It runs automated regex scrubbers that replace:
  - Any email with `[REDACTED_EMAIL]`
  - Any Indian or global phone number with `[REDACTED_PHONE]`
  - Any 13-to-19 digit card PAN with `[REDACTED_CARD]`
  
  The internal system still has the real data to process the payment, but the outside AI model only sees clean, sanitized diagnostic numbers.

---

## 9. Human-in-the-Loop (HITL)

### When and Why Humans Get Involved
Total automation is fantastic for small, everyday orders. If a ₹450 food delivery order fails, an automated retry is harmless. But if an automated bot makes an error on a **₹14,500** luxury watch or a **₹50,000** B2B software license, the merchant could lose significant money or anger a VIP client.

That is why RecoverAI draws a hard line: **any failed order of ₹8,500 or more is paused automatically.**

```text
               ₹14,500 ORDER FAILS
                        │
                        ▼
             Policy verdict: ESCALATE
                        │
                        ▼
      Inserted into 'approval_requests' table
            (Status: PENDING_APPROVAL)
                        │
                        ▼
          MERCHANT DASHBOARD (Tab 4)
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   Click "✓ APPROVE"           Click "✗ REJECT"
          │                           │
          ▼                           ▼
  Razorpay link sent           Order closed
   to customer safely.         Zero action taken.
```

### What the Merchant Sees and Does
When an order escalates, it appears on **Tab 4 (Merchant Approval Queue)** in the Streamlit app:
1. **The Merchant Inspects:**
   - The Order Amount (e.g., ₹14,500.00).
   - The ML Recovery Probability (e.g., 76.7%).
   - The AI Doctor's diagnosis explaining the failure.
   - The exact policy reason it paused (e.g., *"Amount ₹14,500 >= ₹8,500 threshold"*).
2. **The Security Banner:** At the top of the screen, a banner states: *"🛡️ Fraud blocks cannot be overridden."* If an order was blocked for fraud, it cannot be approved here.
3. **The Actions:**
   - Clicking **`✓ APPROVE`** grabs the idempotency lock, calls Razorpay's API to generate the payment link, and marks the request as `APPROVED_BY_HUMAN`.
   - Clicking **`✗ REJECT`** marks the request as `REJECTED_BY_MERCHANT`. The order is permanently closed, and zero payment actions are taken.

---

## 10. Two Full Worked Examples

### Walkthrough 1: The Successful Recovery (ACT)
*Let's follow transaction **`txn_0000012`** (Scenario A: A ₹2,500 T-Shirt purchase with a momentary network timeout).*

1. **Failure Ingestion (`agent/nodes/context.py`):**
   - The checkout fails. The function `load_context()` pulls the transaction record: Amount = ₹2,500, Method = Credit Card, Failure = `network_timeout`.
   - Telemetry shows: Customer has an account for 180 days, 11 successful past orders, 0 recent failures, and a clean IP risk score of `0.04`.
2. **ML Scoring (`agent/nodes/prediction.py` $\to$ `ml/predict.py`):**
   - The 31 features are passed to `recovery_model.joblib`.
   - The model calculates probability: **$P = 0.9855$ (98.55%)**.
   - Expected Recovery Value is calculated: $0.9855 \times ₹2,500 = \mathbf{₹2,463.75}$.
3. **AI Diagnosis (`agent/nodes/diagnosis.py` $\to$ `agent/services/llm_service.py`):**
   - PII redaction runs first, scrubbing names and emails.
   - The LLM reviews the sanitized context and outputs:
     `{"failure_category": "transient", "diagnosis": "Temporary gateway timeout. Highly viable for recovery."}`
4. **Recommendation (`agent/nodes/recommendation.py`):**
   - Strategy engine recommends sending a `payment_link` (Action cost: ₹12.00).
   - Net Expected Value: $₹2,463.75 - ₹12.00 = \mathbf{+₹2,451.75}$ (Massive positive ROI).
5. **Policy Evaluation (`agent/nodes/policy.py`):**
   - The function `evaluate_transaction_policy()` checks:
     - Fraud check: IP risk $0.04 \le 0.70$ (PASSED)
     - Defect check: Not expired/invalid (PASSED)
     - Streak check: Streak $0 < 4$ (PASSED)
     - Viability check: Probability $0.9855 \ge 0.35$ (PASSED)
     - High-value check: Amount $₹2,500 < ₹8,500$ (PASSED)
   - Verdict: **`ACT`** (Action approved for autonomous execution).
6. **Execution (`payment/executor.py`):**
   - Acquires the `_LOCAL_EXECUTION_LOCK`.
   - Inserts claim into `PaymentExecutionClaim` database table.
   - Calls Razorpay Test API: Creates payment link (`https://rzp.io/i/xxxx`).
   - Transaction status in database: **`PROCESSING`** (Not recovered yet!).
7. **Webhook Settlement (`payment/webhook.py` $\to$ `database/repository.py`):**
   - The customer clicks the link on their phone and pays.
   - Razorpay sends a `payment_link.paid` webhook to our server.
   - Our server verifies the HMAC-SHA256 signature. Signature matches!
   - Database record updates: `payment_status = "RECOVERED"`. The revenue is saved!

---

### Walkthrough 2: The Rational Refusal (REFUSE)
*Let's follow transaction **`txn_0000005`** (Scenario C: A ₹7,500 purchase by a suspicious bot).*

1. **Failure Ingestion (`agent/nodes/context.py`):**
   - The checkout fails. The function `load_context()` pulls the transaction: Amount = ₹7,500, Method = Credit Card, Failure Reason = `suspected_risk`.
   - Telemetry shows: IP risk score is **0.92** (Extreme danger), and Velocity score is **0.85** (Bot rapidly submitting cards).
2. **ML Scoring (`agent/nodes/prediction.py`):**
   - The model flags risk level as **`HIGH`** due to IP risk $> 0.60$.
3. **AI Diagnosis (`agent/nodes/diagnosis.py`):**
   - The diagnostic node identifies a high-risk security violation.
4. **Policy Evaluation (`agent/nodes/policy.py`):**
   - The policy engine runs Stage 1 (Refusal Checks):
     ```python
     if ip_risk > 0.70 or reason in ["suspected_risk"]:
         return "REFUSE", "HARD_FRAUD_BLOCK", "no_action"
     ```
   - IP risk (0.92) is far above the 0.70 ceiling. Failure reason is `suspected_risk`.
   - Verdict: **`REFUSE`** (Reason: `HARD_FRAUD_BLOCK`).
5. **Execution Routing (`agent/graph.py`):**
   - The conditional router `route_after_policy()` sees `REFUSE`.
   - It completely skips the payment executor and jumps directly to `create_audit_log`.
6. **Audit Recording (`agent/nodes/audit.py`):**
   - The refusal is recorded in `decision_audit.jsonl` and database `AuditLog`.
   - **Final Result:** Zero payment links created. Zero API calls made. Zero money spent. The order is blocked from human override to prevent chargebacks.

---

## 11. Demo Video Script Notes (Talking Points)

Use these exact talking points when recording your screen demo:

### 🟢 Scenario A — Smart Auto-Recovery (₹2,500 Transient Network Timeout)
* **What to click on screen:** On Tab 2 (Demo Simulator), click **Scenario A**, click the red button **`Simulate Payment Failure & Run Agent`**, then switch to Tab 3 (Live Decision Trace).
* **What to say (2–3 sentences):**
  > *"Here's a normal checkout failure: a loyal customer trying to buy a ₹2,500 item, but their bank has a 2-second timeout. Our ML model evaluates 31 data points and predicts a 98.9% recovery probability. Because the Expected Value is heavily positive and the amount is under our safety threshold, the policy engine issues an immediate green ACT verdict and generates a Razorpay payment link."*
* **The moment to pause on:** Point your cursor at the green **`ACT`** badge in Tab 3, then scroll down and click **`Simulate Customer Paid Webhook Event`** to show the badge turn to **`RECOVERED`**.

### 🟡 Scenario B — High-Value Human Approval (₹14,500 Order)
* **What to click on screen:** On Tab 2, click **Scenario B** (₹14,500), click simulate, notice the yellow **`ESCALATE`** badge, then switch to Tab 4 (Merchant Approval Queue).
* **What to say (2–3 sentences):**
  > *"Now, what happens on a big order? Here is a ₹14,500 purchase. Even though the ML model predicts a solid 76% recovery chance, our policy engine enforces a hard rule: any order of ₹8,500 or more cannot be handled autonomously. It automatically halts and routes the transaction to Tab 4, our Merchant Approval Queue, where a human merchant can review the AI diagnosis and safely click 'Approve'."*
* **The moment to pause on:** Click **`✓ APPROVE`** in Tab 4, and point out the banner at the top: *"Fraud blocks cannot be overridden."*

### 🔴 Scenario C — Fraud / Risk Block (₹7,500 Suspected Risk)
* **What to click on screen:** On Tab 2, click **Scenario C**, click simulate, and point out the bright red **`REFUSE`** badge.
* **What to say (2–3 sentences):**
  > *"In Scenario C, an order fails with a 0.92 IP risk score—indicating a stolen card or bot. A naive retry system would blindly retry this and cause a dispute chargeback. RecoverAI's policy engine detects the fraud telemetry and slaps on a hard REFUSE. Zero API calls are made, zero money moves, and the transaction is permanently prohibited from entering the human approval queue."*
* **The moment to pause on:** Open the policy check drawer showing **`HARD_FRAUD_BLOCK`** with zero actions taken.

### ⚪ Scenario D — Rational Refusal / Negative EV (₹3,200 Axis Decline)
* **What to click on screen:** On Tab 2, click **Scenario D**, click simulate, and show the low probability score.
* **What to say (2–3 sentences):**
  > *"Finally, Scenario D shows economic discipline. A ₹3,200 payment failed via Netbanking for a customer with a 0% past success rate. The ML model calculates just a 5.1% chance of recovery. Because spending fees to send links produces negative Expected Value, the system rationally refuses to intervene, saving the merchant money and sparing the customer spam."*
* **The moment to pause on:** Point at the **5.1%** probability score and the negative expected value explanation.

---

## 12. Likely Questions and How to Answer Them

#### Q1: "Why do you have two model files in the repository (`recovery_model.joblib` and `exp_0_baseline.joblib`)?"
* **Answer:** *"They serve two completely different purposes. `recovery_model.joblib` is our 31-feature production model used by the live agent to make real-time recovery predictions. `exp_0_baseline.joblib` is a frozen 5-feature baseline model used for offline economic benchmarking. We keep the baseline frozen so we can scientifically prove that our 0.35 Expected Value threshold delivers a verified +₹30,774.87 uplift over a standard 0.50 cutoff on identical holdout data."*

#### Q2: "What happens if OpenAI or Groq goes completely offline during a recovery?"
* **Answer:** *"The recovery pipeline does not break. In `agent/services/llm_service.py`, we built a deterministic rule-based fallback heuristic. If the cloud LLM times out or returns a rate limit error (HTTP 429), our code catches the exception and immediately falls back to built-in rule-based diagnosis schemas with zero downtime."*

#### Q3: "Can a merchant accidentally approve a fraudulent transaction from the human queue?"
* **Answer:** *"No, it is mathematically impossible in our code. In `agent/nodes/policy.py`, fraud checks happen in Stage 1 as hard refusals (`REFUSE`). Hard refusals bypass the approval queue entirely. Only transactions that are completely clean of fraud flags but need human oversight due to order size (₹8,500+) ever reach the merchant queue."*

#### Q4: "Why did you pick 0.35 as your recovery cutoff instead of the standard 0.50?"
* **Answer:** *"In payment recovery, the financial payoffs are asymmetric. Sending an automated payment link only costs ₹12, but recovering a payment brings back thousands of rupees. By sweeping thresholds on our development data, we found that lowering the cutoff from 0.50 to 0.35 captured 36 additional recoverable orders, unlocking +₹30,774.87 in incremental net recovery value for an extra action cost of just ₹72."*

#### Q5: "How do you prevent double-charging if an admin clicks 'Approve' twice quickly?"
* **Answer:** *"We use two layers of idempotency defense. First, an in-memory thread lock (`_LOCAL_EXECUTION_LOCK`) catches duplicate clicks within the same process. Second, the `PaymentExecutionClaim` database table enforces a unique primary key constraint on `idempotency_key`. The second click is rejected by the database and simply returns the existing payment link without making another Razorpay API call."*

#### Q6: "Why do you need HMAC-SHA256 signature verification on webhooks?"
* **Answer:** *"Webhooks are public URLs on the internet. Without signature verification, anyone could send a fake POST request pretending to be Razorpay and claiming an order was paid. Razorpay signs every webhook with our private secret key. Our server recomputes the HMAC-SHA256 hash across the raw request body and verifies it matches the header before updating any database record."*

#### Q7: "Is your causal lift estimate statistically significant?"
* **Answer (Honest):** *"Our Propensity Score Inverse Probability Weighting (IPW) analysis in `evaluation/causal_lift.py` estimates an average treatment effect of +1.31 percentage points. However, the 95% bootstrap confidence interval is [-2.50%, +5.56%]. Because it crosses zero, the effect is not statistically distinguishable from zero on this offline synthetic benchmark. Proving definitive causal lift requires a live randomized A/B trial."*

#### Q8: "Why does the LLM not have direct tool-calling access to trigger payments?"
* **Answer:** *"Giving an LLM direct API execution authority in financial systems creates massive liability. LLMs can hallucinate parameters or fail during edge-case errors. We follow a strict separation of concerns: the LLM acts purely as a diagnostic advisor, while execution authority remains 100% behind deterministic code in the policy engine."*

#### Q9: "How do you ensure customer credit card numbers don't leak to OpenAI or Groq?"
* **Answer:** *"In `agent/services/pii_redaction.py`, before any transaction data is turned into an LLM prompt, we make an isolated deep copy and run automated regex scrubbers. Customer emails, phone numbers, and 16-digit card numbers are masked into `[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, and `[REDACTED_CARD]`. The external AI never sees private credentials."*

#### Q10: "What does the 158 test count mean?"
* **Answer:** *"Our automated Pytest suite runs 158 automated tests across 21 test files with a 100% pass rate. These tests continuously verify policy guard boundaries, ML prediction contracts, idempotency concurrency locks, PII redaction patterns, and Razorpay webhook HMAC signatures."*

---

## 13. One-Page Cheat Sheet (Keep beside you!)

```text
================================================================================
                           RECOVERAI POCKET CARD
================================================================================

1. THE CORE FORMULA:
   Expected Recovery Value = (Recovery Probability * Amount) - Action Cost

2. KEY FINANCIAL METRICS (Holdout Test Set N = 633):
   • Total Revenue at Risk:        ₹1,579,773.01
   • Reference Threshold (0.50):   ₹885,790.57 Net Recovery Value (485 saved)
   • EV-Optimal Threshold (0.35):  ₹916,565.44 Net Recovery Value (521 saved)
   • Net Value Uplift:             +₹30,774.87 (+1.95% profit gain)
   • Incremental Action Cost:      Only +₹72.00 to capture +₹30,846.87 gross

3. THE 5 NON-NEGOTIABLE POLICY THRESHOLDS:
   • High-Value Escalation:        >= ₹8,500.00 (Pauses for Human Review)
   • Uncertainty Escalation Band:  0.32 to 0.38
   • Minimum Viability Cutoff:     0.35 (Below this = REFUSE, waste of fees)
   • Max Consecutive Failures:     4 (>= 4 = REFUSE)
   • Fraud IP Risk Limit:          0.70 (> 0.70 = HARD REFUSE, NO OVERRIDE)

4. ACTION COST STRUCTURE:
   • Retry: ₹5.00  |  Reminder: ₹2.00  |  Payment Link: ₹12.00  |  No Action: ₹0.00

5. THE THREE TRAFFIC LIGHT BADGES:
   • 🟢 ACT      : Safe, under ₹8,500, go ahead and send payment link.
   • 🟡 ESCALATE : Over ₹8,500 or uncertain, merchant must click Approve in Tab 4.
   • 🔴 REFUSE   : Fraud, broken card, or low probability. Blocked completely.

6. THE TWO MODEL ARTIFACTS:
   • ml/models/recovery_model.joblib           -> 31 features (Live Runtime Model)
   • ml/models/experiments/exp_0_baseline.joblib -> 5 features (Offline Benchmark)

7. TOP 4 SAFETY INVARIANTS TO SAY:
   • "The LLM advises; it NEVER touches money or calls payment APIs."
   • "Creating a payment link is NOT recovered revenue. Webhooks verify real settlement."
   • "Fraud blocks are hard-coded and can NEVER be overridden by human approval."
   • "158 automated tests passing (100% pass rate)."
================================================================================
```
