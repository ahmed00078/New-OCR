

python main.py process application/data/c07524446.pdf --prompt "Extract the following information from the sustainability report:
- Manufacturer (company/manufacturer name)
- Year (of the report or product)
- Product name (exact model)
- Carbon impact (in kg CO2 eq or equivalent)
- Maximum power consumption (in W, kW, or equivalent)
- Product weight (in kg, g, or equivalent)
Return a JSON with these exact keys: manufacturer, year, product_name, carbon_impact, power_consumption, product_weight"