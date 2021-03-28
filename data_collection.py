from scripts.CreditCardNumbersGenerator import *
from scripts.RandomTCGenerator import *

import numpy as np
import pandas as pd
from phone_gen import PhoneNumber


def credit_card_data():
    mastercard = credit_card_number(
        generator, mastercardPrefixList, 16, 10, (4, 4, 4, 4)
    )
    visa16 = credit_card_number(generator, visaPrefixList, 16, 10, (8, 4, 4))
    visa13 = credit_card_number(generator, visaPrefixList, 13, 10, (4, 4, 5))
    amex = credit_card_number(generator, amexPrefixList, 15, 10, (4, 6, 5))
    discover = credit_card_number(generator, discoverPrefixList, 16, 10, (4, 4, 4, 4))
    diners = credit_card_number(generator, dinersPrefixList, 14, 10, (4, 6, 4))
    enRoute = credit_card_number(generator, enRoutePrefixList, 15, 10, (4, 7, 4))
    jcb = credit_card_number(generator, jcbPrefixList, 16, 10, (4, 4, 4, 4))
    voyager = credit_card_number(generator, voyagerPrefixList, 15, 10, (4, 5, 6))

    credit_card_data = pd.DataFrame(
        np.array(
            [mastercard, visa16, visa13, amex, discover, diners, enRoute, jcb, voyager]
        ).flatten()
    )
    credit_card_data["label"] = "Credit Card Number"

    return credit_card_data


def phone_number_data():
    phone_number_data = []
    for i in range(100):
        phone_number = PhoneNumber("TR").get_number(full=True)
        if i < 30:
            phone_number_data.append(phone_number)
        elif i < 60:
            phone_number_data.append(phone_number[2:])
        else:
            phone_number_data.append(phone_number[3:])

    phone_number_data = pd.DataFrame(phone_number_data)
    phone_number_data["label"] = "Phone number"

    return phone_number_data


def blood_type_data():
    blood_type_data = pd.read_csv("data/data/blood_types.txt", delimiter="\n", header=None)
    blood_type_data["label"] = "Blood type"

    return blood_type_data


def tc_data():
    tc_data = pd.DataFrame([rastgele_tc() for i in range(100)])
    tc_data["label"] = "Republic of Turkey Identity"

    return tc_data


def amount_data():
    amount_data = pd.read_csv("data/data/amount.txt", delimiter="\n", header=None)
    amount_data["label"] = "Amount"

    return amount_data


def hobby_data():
    hobby_data = pd.read_csv("data/data/hobbies.txt", delimiter="/n", header=None)
    hobby_data["label"] = "Hobbies"

    return hobby_data


if __name__ == "__main__":
    print("Collecting data...")
    data = (
        pd.concat(
            [
                phone_number_data(),
                tc_data(),
                hobby_data(),
                amount_data(),
                credit_card_data(),
                blood_type_data(),
            ]
        )
        .reset_index(drop=True)
        .rename(columns={0: "Text"})
    )
    data.astype(str).to_excel("data/train/train_data.xlsx", index=False)
    print("Data saved to 'data/train/train_data.xlsx'")