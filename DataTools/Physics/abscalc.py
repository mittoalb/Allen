import numpy as np
import xraylib
import sys
import argparse

def get_mass_attenuation_coefficients(elements, energy):
    """
    Retrieve the mass attenuation coefficients for each element at the given energy.

    Parameters:
    elements (dict): Dictionary of elements and their quantities in the formula.
    energy (float): X-ray energy in keV.

    Returns:
    dict: Dictionary of elements and their mass attenuation coefficients (cm²/g).
    """
    macs = {}
    for element in elements:
        try:
            macs[element] = xraylib.CS_Total(xraylib.SymbolToAtomicNumber(element), energy)
        except ValueError as e:
            print(f"Error for element {element} at energy {energy} keV: {e}")
            return None
    return macs

def parse_formula(formula):
    """
    Parse the chemical formula into its constituent elements and their quantities.

    Parameters:
    formula (str): Chemical formula (e.g., H2O, C6H12O6).

    Returns:
    dict: Dictionary of elements and their quantities.
    """
    import re
    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    parsed = pattern.findall(formula)
    elements = {}
    for (element, count) in parsed:
        if count == '':
            count = 1
        elements[element] = elements.get(element, 0) + int(count)
    return elements

def calculate_linear_attenuation_coefficient(formula, density, energy):
    """
    Calculate the linear attenuation coefficient for a given formula, density, and X-ray energy.

    Parameters:
    formula (str): Chemical formula (e.g., H2O, C6H12O6).
    density (float): Density of the material in g/cm³.
    energy (float): X-ray energy in keV.

    Returns:
    float: Linear attenuation coefficient in cm⁻¹.
    """
    elements = parse_formula(formula)
    macs = get_mass_attenuation_coefficients(elements, energy)
    if macs is None:
        print("Failed to retrieve mass attenuation coefficients for some elements.")
        return None
    
    total_weight = sum(fraction * xraylib.AtomicWeight(xraylib.SymbolToAtomicNumber(element)) for element, fraction in elements.items())
    
    mu_rho = 0.0
    for element, fraction in elements.items():
        atomic_weight = xraylib.AtomicWeight(xraylib.SymbolToAtomicNumber(element))
        weight_fraction = (fraction * atomic_weight) / total_weight
        mu_rho += weight_fraction * macs[element]
    
    # mu_rho is in cm²/g, multiplying by density (g/cm³) gives cm⁻¹
    linear_attenuation_coefficient = mu_rho * density
    return linear_attenuation_coefficient

def main():
    """
    Main function to parse command-line arguments and calculate the linear attenuation coefficient.
    """
    parser = argparse.ArgumentParser(description='Calculate linear attenuation coefficient for a given formula, density, and X-ray energy.')
    parser.add_argument('--formula', required=True, type=str, help='Chemical formula (e.g., H2O, C6H12O6)')
    parser.add_argument('--density', required=True, type=float, help='Density in g/cm^3')
    parser.add_argument('--energy', required=True, type=float, help='X-ray energy in keV')

    args = parser.parse_args()

    formula = args.formula
    density = args.density
    energy = args.energy

    if not (1 <= energy <= 10000):  # Example range, adjust based on xraylib documentation
        print("Energy out of range. Please provide an energy between 1 keV and 10000 keV.")
        sys.exit(1)

    linear_attenuation_coefficient = calculate_linear_attenuation_coefficient(formula, density, energy)
    
    if linear_attenuation_coefficient is not None:
        print(f'Linear attenuation coefficient for {formula} at {energy} keV with density {density} g/cm^3 is: {linear_attenuation_coefficient:.6f} cm^-1')

if __name__ == '__main__':
    main()

