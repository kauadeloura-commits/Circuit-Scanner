"""Logical component pins, with no image-processing dependency."""
PIN_SCHEMAS = {
    'resistor':['1','2'], 'capacitor':['1','2'], 'capacitor_polarized':['+','-'],
    'inductor':['1','2'], 'diode':['A','K'], 'led':['A','K'],
    'source_dc':['+','-'], 'battery':['+','-'], 'gnd':['GND'], 'vcc':['VCC'],
    'transistor_npn':['B','C','E'], 'transistor_pnp':['B','C','E'],
    'mosfet_n':['G','D','S'], 'mosfet_p':['G','D','S'], 'opamp':['+','-','OUT'],
}
