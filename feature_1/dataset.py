import pandas as pd

dataset = pd.DataFrame(
    [
        {
            "inputs": {"description": "Website hosting fees - AWS", "amount": 145.50},
            "expectations": {
                "expected_response": "GST on Expenses",
                "expected_facts": "Business expense for taxable services with GST included",
            },
        },
        {
            "inputs": {"description": "Consultation fee - client ABC", "amount": 2500.00},
            "expectations": {
                "expected_response": "GST on Income",
                "expected_facts": "Professional services income subject to GST",
            },
        },
        {
            "inputs": {"description": "Fresh fruit and vegetables", "amount": 87.30},
            "expectations": {
                "expected_response": "GST Free Expenses",
                "expected_facts": "Basic food items are GST-free",
            },
        },
        {
            "inputs": {"description": "Export of software services to USA", "amount": 5000.00},
            "expectations": {
                "expected_response": "GST Free Income",
                "expected_facts": "Exports are GST-free supplies",
            },
        },
        {
            "inputs": {"description": "Office supplies - Officeworks", "amount": 234.67},
            "expectations": {
                "expected_response": "GST on Expenses",
                "expected_facts": "Standard business expense with GST",
            },
        },
        {
            "inputs": {"description": "Bank account fees", "amount": 15.00},
            "expectations": {
                "expected_response": "BAS Excluded",
                "expected_facts": "Financial supplies are input taxed, not subject to GST",
            },
        },
        {
            "inputs": {"description": "Freight charges from China", "amount": 450.00},
            "expectations": {
                "expected_response": "GST on Imports",
                "expected_facts": "Imported goods subject to GST at the border",
            },
        },
        {
            "inputs": {"description": "Medical consultation", "amount": 95.00},
            "expectations": {
                "expected_response": "GST Free Expenses",
                "expected_facts": "Health services are GST-free",
            },
        },
        {
            "inputs": {"description": "Rent received - commercial property", "amount": 3200.00},
            "expectations": {
                "expected_response": "GST on Income",
                "expected_facts": "Commercial property rent includes GST",
            },
        },
        {
            "inputs": {"description": "Electricity bill - office", "amount": 387.45},
            "expectations": {
                "expected_response": "GST on Expenses",
                "expected_facts": "Utilities for business use include GST",
            },
        },
        {
            "inputs": {"description": "Interest earned on business account", "amount": 23.50},
            "expectations": {
                "expected_response": "BAS Excluded",
                "expected_facts": "Interest income is input taxed",
            },
        },
        {
            "inputs": {"description": "Sale of software licenses - domestic", "amount": 1800.00},
            "expectations": {
                "expected_response": "GST on Income",
                "expected_facts": "Domestic sale of taxable supplies",
            },
        },
        {
            "inputs": {"description": "Bread and milk", "amount": 12.50},
            "expectations": {
                "expected_response": "GST Free Expenses",
                "expected_facts": "Basic food items are GST-free",
            },
        },
        {
            "inputs": {"description": "Insurance premium - business", "amount": 567.00},
            "expectations": {"expected_response": "BAS Excluded", "expected_facts": "Insurance is input taxed"},
        },
        {
            "inputs": {"description": "Marketing services - Google Ads", "amount": 890.00},
            "expectations": {
                "expected_response": "GST on Expenses",
                "expected_facts": "Advertising expense with GST",
            },
        },
        {
            "inputs": {"description": "Medical supplies imported from Germany", "amount": 1250.00},
            "expectations": {
                "expected_response": "GST on Imports",
                "expected_facts": "Imported medical equipment subject to GST",
            },
        },
        {
            "inputs": {"description": "Education course fees", "amount": 450.00},
            "expectations": {
                "expected_response": "GST Free Expenses",
                "expected_facts": "Educational courses are GST-free",
            },
        },
        {
            "inputs": {"description": "Consulting income - overseas client", "amount": 3500.00},
            "expectations": {
                "expected_response": "GST Free Income",
                "expected_facts": "Services exported to overseas client are GST-free",
            },
        },
        {
            "inputs": {"description": "Laptop purchase for business", "amount": 1899.00},
            "expectations": {
                "expected_response": "GST on Expenses",
                "expected_facts": "Capital equipment purchase includes GST",
            },
        },
        {
            "inputs": {"description": "Residential rent received", "amount": 2400.00},
            "expectations": {
                "expected_response": "BAS Excluded",
                "expected_facts": "Residential rent is input taxed",
            },
        },
    ]
)
