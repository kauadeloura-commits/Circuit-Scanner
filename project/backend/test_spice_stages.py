import copy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from backend.spice_netlist import build_netlist,value_number,analysis_options
from backend.spice_simulation import run_simulation,locate_ngspice,parse_raw


def divider():
    return dict(schema_version='circuit-1.0',status='confirmed',confirmed_by_user=True,
        components=[dict(id='VCC1',type='source_dc',value='5V',pins=['+','-']),
                    dict(id='R_top',type='resistor',value='1kΩ',pins=['1','2']),
                    dict(id='R_bottom',type='resistor',value='1000',pins=['1','2'])],
        nets=[dict(name='VIN',members=['VCC1.+','R_top.1']),dict(name='VOUT',members=['R_top.2','R_bottom.1']),
              dict(name='GND',members=['VCC1.-','R_bottom.2'])])


class SpiceNetlistTests(unittest.TestCase):
    def test_validated_units_si_and_decimal_comma(self):
        for value,kind,expected in [('10kΩ','resistor',10000),('4k7','resistor',4700),('4R7','resistor',4.7),
                                  ('2M2','resistor',2.2e6),('1M','resistor',1e6),('1m','resistor',.001),
                                  ('1meg','resistor',1e6),('4,7nF','capacitor',4.7e-9),('100µF','capacitor',1e-4),
                                  ('100μF','capacitor',1e-4),('10mH','inductor',.01),('-12V','source_dc',-12),
                                  ('1e3','resistor',1000),('0V','source_dc',0)]:
            with self.subTest(value=value):self.assertAlmostEqual(value_number(value,kind),expected)
        for value,kind,unit,expected in [('100uF','capacitor','farad',1e-4),('10mH','inductor','henry',.01),
                                         ('5V','source_dc','volt',5),('1k','resistor','ohm',1000)]:
            with self.subTest(unit=unit):self.assertAlmostEqual(value_number(value,kind,unit),expected)

    def test_values_and_options_cannot_add_spice_commands(self):
        for value in ('1k\n.include x','1k; quit','1e309','nan','0','-10','1kV','10foobar','1e-25'):
            with self.subTest(value=value),self.assertRaises(ValueError):value_number(value,'resistor')
        for opts in ({'analysis':'shell'},{'analysis':'tran','stop':'1','step':'1n'}, {'analysis':'tran','initial_zero':'true'}, {'filename':'x'}):
            with self.assertRaises(ValueError):analysis_options(opts)

    def test_reference_and_net_names_preserved_in_mapping_but_not_executed(self):
        c=divider();c['nets'][0]['name']='unsafe .include x'
        result=build_netlist(c)
        self.assertEqual(result['nodes']['GND'],'0')
        self.assertNotIn('unsafe',result['netlist']);self.assertNotIn('.include',result['netlist'])
        self.assertEqual({b['component_id'] for b in result['components']},{'VCC1','R_top','R_bottom'})
        self.assertIn('DC 5',result['netlist'])

    def test_polarity_maps_plus_and_anode_before_minus_and_cathode(self):
        c=divider();c['components'][2].update(type='capacitor_polarized',pins=['+','-'],value='10uF')
        c['nets'][1]['members'][-1]='R_bottom.+';c['nets'][2]['members'][-1]='R_bottom.-'
        result=build_netlist(c);b=next(b for b in result['components'] if b['component_id']=='R_bottom')
        self.assertEqual(b['nodes']['-'],'0');self.assertNotEqual(b['nodes']['+'],'0')

    def test_unconfirmed_unsupported_missing_ground_and_vcc_only_are_rejected(self):
        for mutate in (lambda c:c.update(confirmed_by_user=False),lambda c:c['components'][0].update(type='opamp',pins=['+','-','OUT']),
                       lambda c:c['nets'][-1].update(name='RETURN'),lambda c:c['components'][0].update(type='vcc',pins=['VCC'])):
            c=divider();mutate(c)
            with self.assertRaises(ValueError):build_netlist(c)


class NgspiceSimulationTests(unittest.TestCase):
    def test_inductor_manual_initial_current_matches_exponential_decay(self):
        c=divider();c['components'][0]['value']='0V';c['components'][2].update(type='inductor',value='1mH',metadata={'electrical_parameters':{'initial_current':.002}})
        data=self.simulate(c,dict(analysis='tran',step='10n',stop='2u',initial_zero=True))
        sample=min(data['samples'],key=lambda s:abs(s['time_s']-1e-6))
        self.assertAlmostEqual(sample['currents_a']['R_bottom'],.002*math.exp(-sample['time_s']/1e-6),delta=5e-6)
    def test_mosfet_lambda_and_gate_capacitances_reach_the_simulator(self):
        from backend.production_validation import reference_cases
        c=copy.deepcopy(next(c for c in reference_cases() if c['id']=='T09')['circuit'])
        initial=self.simulate(c)['last']['voltages_v']['OUT']
        params=c['components'][-1]['metadata']['electrical_parameters'];params.update(channel_lambda=.1,gate_source_capacitance=20e-12,gate_drain_capacitance=7e-12)
        deck=build_netlist(c)['netlist'];self.assertIn('LAMBDA=0.1',deck);self.assertIn('CM1GD',deck);self.assertIn('CM1GS',deck)
        changed=self.simulate(c)['last']['voltages_v']['OUT'];self.assertLess(changed,initial)
    def simulate(self,circuit,options=None):
        if not locate_ngspice():self.skipTest('ngspice portátil não disponível')
        with TemporaryDirectory() as tmp:
            data=run_simulation(build_netlist(circuit,options),tmp)
            self.assertTrue(data['success'],data)
            self.assertTrue((Path(tmp)/'simulation.csv').exists())
            self.assertTrue((Path(tmp)/'simulation.raw').exists())
            self.assertEqual(json.loads((Path(tmp)/'simulation.json').read_text(encoding='utf8'))['point_count'],data['point_count'])
            return data

    def test_voltage_divider_matches_calculation(self):
        data=self.simulate(divider());last=data['last']
        self.assertAlmostEqual(last['voltages_v']['VIN'],5,places=8)
        self.assertAlmostEqual(last['voltages_v']['VOUT'],2.5,places=8)
        self.assertAlmostEqual(last['currents_a']['R_top'],.0025,places=8)
        self.assertAlmostEqual(last['currents_a']['VCC1'],-.0025,places=8)

    def test_rc_charging_matches_time_constant(self):
        c=divider();c['components'][2].update(type='capacitor',value='1uF')
        data=self.simulate(c,dict(analysis='tran',step='10u',stop='5m',initial_zero=True))
        self.assertGreater(data['point_count'],50)
        self.assertAlmostEqual(data['last']['voltages_v']['VOUT'],5*(1-math.exp(-5)),delta=.004)
        sample=min(data['samples'],key=lambda s:abs(s['time_s']-.001))
        self.assertAlmostEqual(sample['voltages_v']['VOUT'],5*(1-math.exp(-sample['time_s']/.001)),delta=.015)

    def test_capacitor_dc_is_open_and_inductor_dc_is_short(self):
        for kind,value,expected in [('capacitor','1u',5.),('inductor','10m',0.)]:
            c=divider();c['components'][2].update(type=kind,value=value)
            data=self.simulate(c)
            self.assertAlmostEqual(data['last']['voltages_v']['VOUT'],expected,places=6)

    def test_diode_and_led_have_coherent_current_and_forward_voltage(self):
        for kind,low,high in [('diode',.5,.85),('led',1.5,2.5)]:
            c=divider();c['components'][2].update(type=kind,value=None,pins=['A','K'])
            c['nets'][1]['members'][-1]='R_bottom.A';c['nets'][2]['members'][-1]='R_bottom.K'
            data=self.simulate(c);last=data['last'];voltage=last['voltages_v']['VOUT']
            self.assertGreater(voltage,low);self.assertLess(voltage,high)
            self.assertAlmostEqual(last['currents_a']['R_bottom'],(5-voltage)/1000,places=7)
            self.assertTrue(data['warnings'])

    def test_missing_engine_timeout_and_bad_process_are_reported(self):
        from subprocess import TimeoutExpired,CompletedProcess
        for effect,code in ((TimeoutExpired('ngspice',45),'timeout'),(OSError('indisponível'),'result_error')):
            with TemporaryDirectory() as tmp,patch('backend.spice_simulation.subprocess.run',side_effect=effect):
                data=run_simulation(build_netlist(divider()),tmp,command='ngspice')
                self.assertFalse(data['success']);self.assertEqual(data['error_code'],code)
        with TemporaryDirectory() as tmp,patch('backend.spice_simulation.locate_ngspice',return_value=None):
            self.assertEqual(run_simulation(build_netlist(divider()),tmp)['error_code'],'simulator_missing')

    def test_truncated_raw_never_produces_success(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.raw';path.write_text('Variables:\n0 v(n1) voltage\nValues:\n0',encoding='ascii')
            with self.assertRaises(ValueError):parse_raw(path)
