"""Build the reference annotations from hand-read values plus located quotations.

The value in each row was decided by reading the article. The *quotation* is
located here: the script finds the sentence containing the given anchor and
checks it is verbatim in the document. Doing it this way means no quotation can
be mistyped, and any annotation whose supporting sentence cannot be found fails
loudly instead of shipping unsupported.
"""

import re
import sys

import pandas as pd

SENT = re.compile(r"(?<=[.!?])\s+")

# (pmcid, field, value, anchor) — anchor is a distinctive substring of the
# sentence that states the value.
ANNOTATIONS = [
    # NoHoW: 2x2 factorial, single-blind, 1627 adults, weight change at 12 months
    ("PMC13189573", "n_randomised", "1627", "1627 adults who had achieved"),
    ("PMC13189573", "design", "factorial", "2×2 factorial randomized"),
    ("PMC13189573", "blinding", "single-blind", "2×2 factorial randomized, single-blind"),
    ("PMC13189573", "primary_outcome", "weight change from baseline", "The primary outcome was weight change"),
    ("PMC13189573", "follow_up_weeks", "52.18", "primary outcome was change in weight (kilograms) at 12 month"),
    ("PMC13189573", "registration_id", "ISRCTN88405328", "ISRCTN88405328"),

    ("PMC13215628", "n_randomised", "20", "randomized crossover trial of 20 participants"),
    ("PMC13215628", "design", "crossover", "randomized crossover trial of 20 participants"),
    ("PMC13215628", "registration_id", "UMIN000052303", "UMIN000052303"),

    ("PMC13225348", "n_randomised", "214", "a total of 214 patients"),
    ("PMC13225348", "design", "parallel-group", "randomized controlled trial that investigates the difference"),
    ("PMC13225348", "follow_up_weeks", "52.18", "weight outcomes at one year postoperatively"),
    ("PMC13225348", "registration_id", "NCT05472922", "NCT05472922"),

    ("PMC13235231", "n_randomised", "12", "Twelve trained cyclists"),
    ("PMC13235231", "design", "crossover", "double blind, randomized, placebo-controlled, cross-over trial"),
    ("PMC13235231", "blinding", "double-blind", "double blind, randomized, placebo-controlled, cross-over trial"),
    ("PMC13235231", "primary_outcome", "time trial performance", "time trial performance (our primary outcome)"),
    ("PMC13235231", "registration_id", "NCT05404841", "Registered with clinicaltrials.gov"),

    ("PMC13268633", "n_randomised", "86", "86 adults living in 2 municipal refugee shelters"),
    ("PMC13268633", "design", "parallel-group", "2-arm, parallel-group randomized controlled trial"),
    ("PMC13268633", "primary_outcome", "plaque index", "Plaque index and gingival index were assessed"),
    ("PMC13268633", "follow_up_weeks", "8.69", "at 2-month follow-up"),
    # The article states assessor blinding could not be achieved, which is an
    # unblinded assessment however it is phrased. Annotating it as open-label is
    # a judgement call and is listed in the README as one.
    ("PMC13268633", "blinding", "open-label", "complete assessor blinding could not be guaranteed"),
    ("PMC13268633", "registration_id", "DRKS00032017", "DRKS00032017"),

    ("PMC13279985", "n_randomised", "28", "were randomised to SEI or SI"),
    ("PMC13279985", "design", "parallel-group", "were randomised to SEI or SI"),
    ("PMC13279985", "primary_outcome", "feasibility: recruitment, retention, compliance", "Feasibility outcomes (recruitment, retention, compliance)"),
    ("PMC13279985", "follow_up_weeks", "14", "14-week, remotely delivered sleep-enhanced intervention"),
    ("PMC13279985", "registration_id", "NCT05273892", "NCT05273892"),

    ("PMC13282192", "n_randomised", "126", "conducted with 126 women experiencing PMS"),
    ("PMC13282192", "design", "parallel-group", "conducted with 126 women experiencing PMS"),
    ("PMC13282192", "primary_outcome", "premenstrual symptom severity", "a significant reduction in premenstrual symptom severity"),
    ("PMC13282192", "registration_id", "NCT06942169", "The trial was retrospectively registered"),

    ("PMC13288546", "n_randomised", "14", "participants ( n = 14) were randomly allocated"),
    ("PMC13288546", "design", "crossover", "complete two EHS"),
    ("PMC13288546", "blinding", "double-blind", "double-blind placebo-controlled design"),
    ("PMC13288546", "registration_id", "12624000285550", "Australian New Zealand Clinical Trials Registry"),

    ("PMC13289930", "design", "parallel-group", "randomised 1:1 to high-intensity interval training"),
    ("PMC13289930", "primary_outcome", "reduction in liver fat-fraction", "The primary outcome was reduction in liver fat-fraction"),
    ("PMC13289930", "follow_up_weeks", "12", "high-intensity interval training over 12 weeks"),
    ("PMC13289930", "blinding", "single-blind", "assessed the primary outcome of liver fat-fraction blinded"),
    ("PMC13289930", "registration_id", "NCT05265026", "NCT05265026"),

    ("PMC13290998", "n_randomised", "42", "42 post-treatment AYA cancer survivors"),
    ("PMC13290998", "design", "parallel-group", "randomized 2:1 to relaxing or sham acupressure"),
    ("PMC13290998", "primary_outcome", "feasibility: recruitment, retention, fidelity, adherence, acceptability", "Feasibility outcomes included recruitment, retention"),
    ("PMC13290998", "follow_up_weeks", "10", "sustained through the 10-week follow-up period"),
    ("PMC13290998", "registration_id", "NCT06442891", "ClinicalTrials.gov Identifier: NCT06442891"),

    ("PMC13291922", "n_randomised", "30", "Thirty sub‐acute stroke patients"),
    ("PMC13291922", "design", "parallel-group", "randomly assigned to the experimental group"),
    ("PMC13291922", "blinding", "double-blind", "using a double‐blind sham‐controlled design"),
    ("PMC13291922", "follow_up_weeks", "2", "5 days/week for 2 weeks"),
    ("PMC13291922", "registration_id", "ChiCTR2200066237", "ChiCTR2200066237"),

    ("PMC13293601", "n_randomised", "317", "159 participants in the immediate ICBT group"),
    ("PMC13293601", "design", "parallel-group", "159 participants in the immediate ICBT group"),
    ("PMC13293601", "blinding", "open-label", "8-week nonblinded pragmatic RCT"),
    ("PMC13293601", "primary_outcome", "depressive symptoms", "greater reduction in depressive symptoms"),
    ("PMC13293601", "follow_up_weeks", "8", "An 8-week randomized controlled trial"),
    ("PMC13293601", "registration_id", "ChiCTR2100046425", "ChiCTR2100046425"),

    ("PMC13296589", "n_randomised", "36", "randomised 36 ICU survivors"),
    ("PMC13296589", "design", "parallel-group", "parallel, three‐arm, assessor‐blinded pilot randomised controlled trial"),
    ("PMC13296589", "blinding", "single-blind", "assessor‐blinded pilot randomised controlled trial"),
    ("PMC13296589", "primary_outcome", "feasibility: recruitment, retention, adherence, and safety", "The primary outcomes were feasibility"),
    ("PMC13296589", "follow_up_weeks", "8", "8‐week blended program"),
    ("PMC13296589", "registration_id", "NCT06117761", "NCT06117761"),
]


def sentence_around(text: str, index: int, length: int) -> str:
    left = 0
    for m in SENT.finditer(text, 0, index):
        left = m.end()
    right_match = SENT.search(text, index + length)
    right = right_match.start() + 1 if right_match else len(text)
    return text[left:right].strip()


def main() -> None:
    df = pd.read_parquet("data/silver/corpus.parquet")
    by_id = {r["document_id"].split(":")[1]: r for _, r in df.iterrows()}

    rows, failures = [], []
    for pmcid, field, value, anchor in ANNOTATIONS:
        row = by_id.get(pmcid)
        if row is None:
            failures.append(f"{pmcid}: not in corpus")
            continue
        text = row["text"]
        index = text.find(anchor)
        if index < 0:
            failures.append(f"{pmcid}/{field}: anchor not found: {anchor!r}")
            continue
        quote = sentence_around(text, index, len(anchor))
        if quote not in text:
            failures.append(f"{pmcid}/{field}: located quote is not verbatim")
            continue
        if len(quote) < 20:
            failures.append(f"{pmcid}/{field}: quote too short ({len(quote)})")
            continue
        rows.append(
            {
                "document_id": f"pmc:{pmcid}",
                "field": field,
                "value": value,
                "quote": quote,
                "section": "",
            }
        )

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  " + f)
        sys.exit(1)

    gold = pd.DataFrame(rows).astype("string")
    keep = sorted({r["document_id"] for r in rows})
    documents = df[df["document_id"].isin(keep)].reset_index(drop=True)

    documents.to_parquet("data/sample/corpus.sample.parquet", index=False)
    gold.to_parquet("data/sample/gold.sample.parquet", index=False)

    print(f"{len(documents)} documents, {len(gold)} reference claims")
    for r in rows:
        print(f"  {r['document_id'][4:]:<12} {r['field']:<18} {r['value'][:34]:<34} | {r['quote'][:70]}")


if __name__ == "__main__":
    main()
