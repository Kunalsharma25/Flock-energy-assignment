# Recommendation: Washing Machine Disaggregation from 1-Minute Smart Meter Data

**Audience:** Utility product and data leadership  
**Source:** REFIT Electrical Load Measurements, 20 UK homes, Oct 2013 – Jul 2015

---

## Executive Summary

A LightGBM model trained on whole-house aggregate power from 17 homes can estimate washing-machine energy with a **42% sum-absolute energy error (SAE) across 4 unseen test homes**, detecting **50% of active washing minutes** and **48% of full wash cycles** in cross-validation. On the primary unseen test home (House 6), SAE is 0.46 and minute-level F1 is 0.20.

**These results support one clear commercial use case: monthly appliance energy breakdowns for customer engagement.** They do not support real-time billing, automated demand-response actuation, or per-cycle precision alerts without further hardware investment.

### Suitable uses
- Estimated monthly washing-machine energy share on customer bills ("Your washing machine used approximately X kWh this month").
- Cohort-level insight into when households do laundry, for load-shifting campaign design.
- Anomaly detection: a household whose estimated washing-machine usage drops to near zero for several weeks may have a broken appliance.

### Unsuitable uses
- **Billing-grade per-cycle metering:** Cycle energy MAPE is 60–74% on cross-validation homes. At this precision, billing customers for individual cycles would be legally and commercially indefensible.
- **Real-time automation:** The model has a cycle precision of 30% on the primary test home (1 in 3 predicted cycles is a true cycle). Triggering smart-plug actions or demand-response signals on this output would cause frequent false activations.
- **Homes unlike the training set:** The model has not been validated on any Indian household data. Indian domestic washing machines are predominantly top-loading semi-automatics with cold-water wash cycles and no resistance heating element. The primary signal the model uses — a sustained 2,000 W heating spike — will be absent, and cross-household transfer performance is unknown.

---

## Section 4 Answers

### 4.1 One Realistic Energy-Saving Opportunity: Peak-Hour Laundry Shift

#### The Opportunity
Our EDA on the four cohort homes (`figures/sec2_usage_patterns.png`) shows a secondary laundry peak between 16:00 and 20:00 local time. A median wash cycle consumes between 267 Wh (House 5) and 673 Wh (House 6). If a utility offers time-of-use pricing with a peak window, nudging customers to shift even a fraction of this peak-hour laundry to off-peak would reduce grid stress.

**Important caveat:** The exact share of peak-hour starts depends on each household's schedule and the specific tariff window chosen. The figure below is illustrative of the methodology, not a UK-wide statistic.

#### Measurement Plan

| Step | Detail |
|:---|:---|
| **Primary metric** | Share of wash cycles starting inside a stated peak window (e.g. 16:00–20:00), measured from smart meter disaggregation or a dedicated smart-plug sub-meter. |
| **Secondary metric** | Total energy shifted out of the peak window (kWh/household/month). |
| **Design** | Randomised controlled trial. Randomly assign households to treatment (receives a nudge notification when a cycle is predicted to start during the peak window) or control (no nudge). Pre-register a minimum detectable effect size (e.g. 5 percentage-point reduction in peak-window starts). |
| **Duration** | Minimum 3 months, ideally spanning at least one seasonal transition to control for summer vs. winter behavioural differences. |
| **Controls** | Household size, tariff type, baseline peak-window start rate, week-of-year. |
| **Caution** | The disaggregation model's 30% cycle precision on unseen homes means some nudge triggers will fire on non-laundry appliances. A short pilot with sub-metered ground truth should validate trigger accuracy before a full rollout. |

---

### 4.2 Limitations

#### Overlapping Appliance Signatures (Thermal Confounders)
A washing machine water-heating element draws 2,000–2,500 W of resistive power. An electric kettle, toaster, oven, and space heater draw the same range. At 1-minute resolution, these are electrically indistinguishable in the aggregate signal. The diagnostic figure `figures/s3_failure_confounder.png` shows a day in House 6 where the model predicted active laundry during periods when a high-wattage kitchen appliance was the actual load. This is the primary driver of low precision (16% on the primary test home).

#### Missing and Flagged Data
Between 1.75% and 7.59% of minutes in the cohort homes carry the REFIT `Issues == 1` flag (appliance sum exceeds aggregate). These minutes are excluded from model training and evaluation targets. Additionally, stale forward-fill periods (1.96% of total data) are masked. On homes with high flag rates, the effective evaluation window is smaller than the nominal date coverage.

#### Household Differences
Each home has a different washing machine brand, cycle programme selection, and occupancy pattern. A model trained on homes with predominantly 60°C hot-wash cycles (detectable 2 kW heating phase) transfers poorly to homes using cold or eco-wash programmes (100–150 W motor only, no heating spike). House 6 in this submission is an example: its median cycle duration is 129 minutes but its typical active power is well below 200 W, making it nearly invisible to a model trained on heating-dominated signatures.

#### UK to India
UK domestic washing machines are predominantly front-loading, automatic, 240 V machines with a 2,000 W internal resistance heater. Indian domestic washing machines are predominantly top-loading, semi-automatic, or bucket-fill machines using cold water only, operating on 220 V with frequent voltage fluctuations. The electrical signature this model has learned — a sustained high-power heating phase followed by motor agitation — may not exist in the Indian market. Before deploying in India, a new labelled dataset of Indian domestic load profiles would be required, and the model architecture would need to be retrained and re-evaluated.

---

### 4.3 What Gets Harder with 15- or 30-Minute Data

A median washing-machine cycle lasts 39–129 minutes across our cohort. At 1-minute resolution this is 39–129 data points per cycle. At 15-minute resolution, the same cycle becomes 3–9 data points; at 30-minute resolution, 2–5 data points.

Several consequences follow:

- **Cycle edge detection fails.** The characteristic heating ramp (2,000 W over ~2 minutes) is averaged into a 15- or 30-minute bin alongside whatever else was running, smearing the onset signature below detectable levels.
- **Lag/lead features lose meaning.** A ±30-minute context window that spans 30 rows at 1-minute resolution spans only 2 rows at 15-minute resolution — insufficient to distinguish a multi-phase cycle from a single large appliance event.
- **Short cycles become undetectable.** House 5's shortest cycles (10–15 minutes) would each occupy less than one 15-minute bin, making detection probabilistically impossible regardless of algorithm.
- **The modelling paradigm shifts.** With 15/30-minute data you move from cycle-level disaggregation to usage-pattern inference: estimating what fraction of the hour's energy came from the washing machine rather than when it ran or how much any single cycle consumed. Metrics would shift from cycle recall and SAE to hourly energy share error.

In short: a 15-minute meter renders washing-machine cycle disaggregation largely intractable. A 30-minute meter makes it entirely intractable with the approach described here.
