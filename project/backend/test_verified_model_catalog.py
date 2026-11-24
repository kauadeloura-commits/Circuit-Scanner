import unittest
from backend.component_models import lookup_model,model_ocr_reading,resolve_parameters,automatic_model_fields


class VerifiedModelCatalogTests(unittest.TestCase):
    def test_each_rectifier_is_registered_as_its_own_code(self):
        for n in range(1,8):
            code='1N400'+str(n)
            item=lookup_model(code,'diode')
            self.assertEqual(item['model'],code)
            self.assertIn('vishay.com/docs/88503/',item['url'])
            self.assertFalse(model_ocr_reading(code)['needs_review'])

    def test_maximum_diode_drop_is_not_inserted_as_a_typical_value(self):
        result=resolve_parameters('diode','1N4007',{},dict(forward_voltage='datasheet',junction_capacitance='datasheet'))
        self.assertNotIn('forward_voltage',result['parameters'])
        self.assertTrue(any(r['parameter']=='forward_voltage' and r['reason']=='only_limit' for r in result['pending']))
        self.assertEqual(result['parameters']['junction_capacitance']['value'],15e-12)

    def test_classic_quad_opamp_uses_its_own_verified_data(self):
        item=lookup_model('LM324','opamp')
        self.assertEqual(item['parameters']['open_loop_gain']['typ'],100000)
        self.assertEqual(item['parameters']['open_loop_gain']['min'],25000)
        self.assertEqual(item['parameters']['gbw']['typ'],1200000)
        self.assertEqual(item['parameters']['slew_rate']['typ'],.5)
        self.assertIn('ti.com/lit/ds/symlink/lm324.pdf',item['url'])

    def test_unregistered_suffix_is_preserved_for_review(self):
        row=model_ocr_reading('LM324B')
        self.assertEqual(row['text'],'LM324B')
        self.assertTrue(row['needs_review'])
        self.assertIsNone(lookup_model('LM324B','opamp'))

    def test_model_identity_never_overrides_a_manual_parameter(self):
        component=dict(type='opamp',model=dict(text='LM324',source='manual'),electrical_parameters=dict(open_loop_gain=dict(value=80000,source='manual')),parameter_policy={})
        result=automatic_model_fields(component)
        self.assertEqual(result['electrical_parameters']['open_loop_gain']['value'],80000)
        self.assertEqual(result['electrical_parameters']['open_loop_gain']['source'],'manual')

    def test_catalog_does_not_guess_different_model_digits(self):
        for text in ('1N4008','BC347'):
            self.assertEqual(model_ocr_reading(text)['text'],text)


if __name__=='__main__':unittest.main()
