import copy,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from backend import ocr
from backend.measurement_text import parse_measurement,unit_anchor
from backend.component_models import model_ocr_reading
from backend.test_component_models import token,detection
from backend.spice_netlist import value_number


def fragment(text,x,engine='easyocr',variant='one',source=None):
    t=dict(text=text,raw_text=text,confidence=.95,bbox=dict(x1=x,y1=30,x2=x+max(6,len(text)*8),y2=45),
           engine=engine,variant=variant,orientation='normal')
    if source:t['source_detection_id']=source
    return t


class UnitFirstOCRTests(unittest.TestCase):
    def test_unit_names_symbols_and_prefixes(self):
        cases=[('10 microfarads','capacitor',10e-6),('2,2 µF','capacitor_polarized',2.2e-6),
               ('.47nF','capacitor',.47e-9),('5 farads','capacitor',5),('4n7F','capacitor',4.7e-9),
               ('10 millihenries','inductor',.01),('2 henry','inductor',2),('10 µH','inductor',1e-5),
               ('4,7 kiloohms','resistor',4700),('1 MΩ','resistor',1e6),('1 mΩ','resistor',.001),
               ('4R7','resistor',4.7),('4k7','resistor',4700),('2.2megΩ','resistor',2.2e6),
               ('9 volts','battery',9),('-5V','source_dc',-5),('500mV','source_dc',.5),
               ('1e-6 F','capacitor',1e-6)]
        for text,kind,expected in cases:
            with self.subTest(text=text):
                result=parse_measurement(text,kind);self.assertIsNotNone(result)
                self.assertAlmostEqual(float(result['normalized']),expected)
                self.assertAlmostEqual(value_number(result['text'],kind,result['unit']),expected)

    def test_units_bind_only_to_matching_classes(self):
        for text,wrong in [('10V','capacitor'),('10uF','inductor'),('10H','resistor'),('10Ω','source_dc')]:
            self.assertIsNone(parse_measurement(text,wrong));self.assertFalse(ocr._parece_valor(text,wrong))
        self.assertEqual(unit_anchor('microfarads'),'farad');self.assertEqual(unit_anchor('kΩ'),'ohm')
        self.assertIsNone(parse_measurement('4R7F','capacitor'))
        self.assertIsNone(parse_measurement('10 16V','source_dc'))

    def test_letter_repairs_require_an_explicit_unit(self):
        for raw,expected in [('lOuF','10uF'),('IOV','10V'),('4SΩ','45Ω'),('2Z0 ohms','220Ω'),('1B0Ω','180Ω')]:
            result=parse_measurement(raw);self.assertEqual(result['text'],expected)
            self.assertEqual(result['raw_text'],raw);self.assertTrue(result['corrections']);self.assertTrue(result['needs_review'])
        self.assertIsNone(parse_measurement('IO','resistor'))
        self.assertEqual(ocr._limpar_texto('81'),'81');self.assertEqual(ocr._limpar_texto('8V'),'8V')
        self.assertEqual(ocr._limpar_texto('RI'),'R1');self.assertEqual(ocr._limpar_texto('BC547B'),'BC547B')

    def test_conflicting_scales_and_decimals_remain_separate_and_flagged(self):
        for a,b in [('1mΩ','1MΩ'),('1.0V','10V'),('10F','10H')]:
            self.assertNotEqual(ocr._chave_leitura(a),ocr._chave_leitura(b))
        self.assertEqual(ocr._chave_leitura('4,7 kiloohms'),ocr._chave_leitura('4.7kΩ'))
        result=ocr.criar_componentes_logicos([token('1mΩ',25),token('1MΩ',25,confidence=.85)],
                                           [detection('A','resistor',20)])
        value=result['components'][0]['value']
        self.assertEqual(value['normalized'],'0.001');self.assertEqual(value['alternatives'],['1mΩ','1MΩ'])
        self.assertIn('numeric_ocr_conflict',result['components'][0]['review_reasons'])

    def test_model_letters_gain_groups_and_unknown_codes_survive(self):
        for raw in ('BC547B','BC547C','BSS138','LM358','IRF540N','SRR6028-100Y'):
            result=model_ocr_reading(raw);self.assertEqual(result['text'],raw);self.assertFalse(result['corrections'])
        self.assertEqual(model_ocr_reading('BCS47')['text'],'BC547')
        self.assertEqual(model_ocr_reading('IRF54ON')['text'],'IRF540N')
        # A genuine suffix is not blindly turned into a digit.
        result=model_ocr_reading('LM35B');self.assertEqual(result['text'],'LM35B');self.assertIn('LM358',result['candidates'])
        self.assertEqual(model_ocr_reading('CUSTOM123')['text'],'CUSTOM123')

    def test_split_number_micro_and_unit_are_joined_without_losing_digits(self):
        for parts,wanted in [([fragment('10',10),fragment('µ',28),fragment('F',38)],'10μF'),
                             ([fragment('4',10),fragment(',',18),fragment('7',24),fragment('µ',32),fragment('F',40)],'4.7uF'),
                             ([fragment('1',10),fragment('0',18),fragment('V',27)],'10V'),
                             ([fragment('81',10),fragment('Ω',28)],'81Ω')]:
            joined=ocr._combinar_tokens_fragmentados(parts)
            self.assertEqual(len(joined),1);self.assertEqual(ocr._normalizar_valor(joined[0]['text'],'source_dc' if wanted.endswith('V') else 'resistor' if wanted.endswith('Ω') else 'capacitor')['text'],wanted.replace('μ','u'))
            self.assertEqual(len(joined[0]['joined_parts']),len(parts))
            self.assertEqual(len(ocr._without_joined_parts(parts+joined)),1)

    def test_model_fragmentation_preserves_the_gain_suffix(self):
        parts=[fragment('BC',10),fragment('547',28),fragment('B',54)]
        joined=ocr._combinar_tokens_fragmentados(parts)
        self.assertEqual([t['text'] for t in joined],['BC547B'])
        result=ocr.criar_componentes_logicos(parts,[detection('A','transistor_npn',20)])
        self.assertEqual(result['components'][0]['model']['text'],'BC547B')
        result=ocr.criar_componentes_logicos([fragment('BC',10),fragment('12345',28)],
                                           [detection('A','transistor_npn',20)])
        self.assertEqual(result['components'][0]['model']['text'],'BC12345')
        self.assertTrue(result['components'][0]['model']['needs_review'])

    def test_no_joining_across_rows_views_or_component_crops(self):
        for parts in ([fragment('10',10,source='A'),fragment('uF',28,source='B')],
                      [fragment('10',10,variant='one'),fragment('uF',28,variant='two')],
                      [fragment('10',10),dict(fragment('uF',28),bbox=dict(x1=28,y1=70,x2=44,y2=85))],
                      [fragment('10',10),fragment('20',28),fragment('V',47)]):
            joined=ocr._combinar_tokens_fragmentados(parts)
            self.assertFalse(any(t['text']=='1020V' for t in joined))
            if len(parts)==2:self.assertEqual(joined,[])

    def test_unit_value_wins_over_a_closer_bare_number(self):
        tokens=[token('999',22),token('4,7 kiloohms',70)]
        result=ocr.criar_componentes_logicos(tokens,[detection('A','resistor',20)])
        self.assertEqual(result['components'][0]['value']['normalized'],'4700')
        self.assertEqual(result['components'][0]['value']['unit'],'ohm')

    def test_low_confidence_unit_does_not_block_a_supported_value(self):
        result=ocr.criar_componentes_logicos([token('10Ω',22,confidence=.01),token('4700',80)],
                                           [detection('A','resistor',20)])
        self.assertEqual(result['components'][0]['value']['normalized'],'4700')
        self.assertTrue(any(t['text']=='10Ω' for t in result['unassigned_tokens']))

    def test_whitelist_applies_only_to_numeric_focus_and_keeps_raw_tsv(self):
        image=np.full((25,60),255,np.uint8)
        tsv='level\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t2\t20\t10\t0\tlOuF\n'
        response=subprocess.CompletedProcess([],0,tsv.encode(),b'')
        with patch.object(ocr.subprocess,'run',return_value=response) as run:
            tokens=ocr._executar_tesseract_tsv(image,'tesseract','test','normal',1,60,25)
            self.assertFalse(any('whitelist' in a for a in run.call_args.args[0]))
            self.assertEqual(tokens[0]['raw_text'],'lOuF')
            ocr._executar_tesseract_tsv(image,'tesseract','test','normal',1,60,25,digits_only=True)
            self.assertIn('tessedit_char_whitelist=0123456789.,+-eE',run.call_args.args[0])

    def test_easyocr_keeps_letters_punctuation_and_original_writing(self):
        from unittest.mock import Mock
        reader=Mock();reader.readtext.return_value=[([[0,0],[40,0],[40,20],[0,20]],'Vth = 2,0V',.95)]
        with patch.object(ocr,'_carregar_easyocr',return_value=reader):
            tokens=ocr._executar_easyocr(np.full((25,60),255,np.uint8),1)
        self.assertNotIn('allowlist',reader.readtext.call_args.kwargs)
        self.assertEqual(tokens[0]['raw_text'],'Vth = 2,0V')
        self.assertIn('=',tokens[0]['text'])

    def test_capacitance_and_voltage_labels_do_not_swap_components(self):
        result=ocr.criar_componentes_logicos([token('100uF',24),token('16V',24,115),token('5V',185)],
                                           [detection('A','capacitor',20),detection('B','source_dc',180)])
        self.assertEqual(result['components'][0]['value']['text'],'100uF')
        self.assertEqual(result['components'][1]['value']['text'],'5V')
        self.assertTrue(any(t['text']=='16V' for t in result['unassigned_tokens']))

    def test_labelled_parameters_join_fragments_and_use_their_units(self):
        parts=[fragment('Vth',10),fragment('=',35),fragment('I.5V',44)]
        result=ocr.criar_componentes_logicos(parts,[detection('A','mosfet_n',20)])
        entry=result['components'][0]['electrical_parameters']['vgs_threshold']
        self.assertEqual(entry['value'],1.5);self.assertEqual(entry['raw_text'],'Vth = I.5V')
        self.assertTrue(entry['needs_review']);self.assertIn('parameter_ocr_corrected',result['components'][0]['review_reasons'])
        self.assertEqual(ocr._limpar_texto('='),'=')
        result=ocr.criar_componentes_logicos([token('Vth=10uF',25)],[detection('A','mosfet_n',20)])
        self.assertNotIn('vgs_threshold',result['components'][0]['electrical_parameters'])

    def test_corrected_value_and_model_keep_original_for_review(self):
        result=ocr.criar_componentes_logicos([token('lOuF',25),token('BCS47',205)],
                                           [detection('A','capacitor',20),detection('B','transistor_npn',200)])
        c,q=result['components'];self.assertEqual(c['value']['text'],'10uF');self.assertEqual(c['value']['raw_text'],'lOuF')
        self.assertTrue(c['value']['corrections']);self.assertIn('numeric_ocr_corrected',c['review_reasons'])
        self.assertEqual(q['model']['text'],'BC547');self.assertEqual(q['model']['raw_text'],'BCS47')
        self.assertIn('model_ocr_needs_confirmation',q['review_reasons'])

    def test_numeric_only_reread_recovers_a_missing_number_near_unit(self):
        image=np.full((100,150,3),255,np.uint8);anchor=fragment('uF',95,engine='tesseract')
        numeric=fragment('10',45,engine='tesseract',variant='unit_numeric_focus')
        with patch.object(ocr,'_executar_tesseract_tsv',return_value=[numeric]) as reread:
            added,raw=ocr._reread_unit_numbers(image,[anchor],'tesseract')
        self.assertTrue(reread.call_args.kwargs['digits_only'])
        self.assertEqual(added[0]['text'],'10uF');self.assertEqual(len(raw),1)

    def test_reread_disagreement_is_preserved_and_models_are_not_whitelisted(self):
        image=np.full((100,200,3),255,np.uint8);anchor=fragment('IOV',40)
        numeric=fragment('18',0,engine='tesseract')
        with patch.object(ocr,'_executar_tesseract_tsv',return_value=[numeric]) as read:
            added,raw=ocr._reread_unit_numbers(image,[anchor,fragment('BC547B',120)],'tesseract')
        self.assertEqual(read.call_count,1);self.assertEqual(added,[])
        self.assertEqual(anchor['numeric_alternatives'],['10V','18V'])

    def test_unknown_writing_and_original_readings_are_archived(self):
        image=np.full((80,180,3),255,np.uint8);raw=[fragment('anotacao',10),fragment('81',85),fragment('Ω',105)]
        with (patch.object(ocr,'easyocr_disponivel',return_value=False),patch.object(ocr,'_executar_tesseract_tsv',return_value=copy.deepcopy(raw)),
              patch.object(ocr,'_detectar_tokens_por_amostras',return_value=[]),patch.object(ocr,'_aplicar_biblioteca_ocr',side_effect=lambda _,tokens,dets:tokens),
              patch.object(ocr,'_reread_unit_numbers',return_value=([],[]))):
            tokens,warnings=ocr.extrair_tokens_ocr(image,'tesseract')
        self.assertEqual(set(t['raw_text'] for t in tokens.raw_tokens),{'anotacao','81','Ω'})
        self.assertEqual(len(tokens.raw_tokens),6)  # Both OCR image variants remain archived.
        self.assertTrue(any(t['text']=='81Ω' for t in tokens))

    def test_saved_json_preserves_raw_writing_and_si_value(self):
        dets=[detection('A','capacitor',20)]
        tokens=ocr.OCRTokenList([token('lOuF',25)]);tokens.raw_tokens=[fragment('lOuF',25),fragment('anotação',100)]
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);photo=folder/'image.png';cv2.imwrite(str(photo),np.full((160,250,3),255,np.uint8))
            with (patch.object(ocr,'localizar_tesseract',return_value='test'),patch.object(ocr,'ocr_status',return_value={'ready':True}),
                  patch.object(ocr,'extrair_tokens_ocr',return_value=(tokens,[])),patch.object(ocr,'salvar_negativos_texto_yolo',return_value=0),
                  patch.object(ocr,'remover_falsos_positivos_textuais',return_value=(dets,[])),patch.object(ocr,'ajustar_caixas_com_texto',return_value=(dets,[])),
                  patch.object(ocr,'filtrar_deteccoes_em_texto',return_value=(dets,[]))):
                result=ocr.executar_ocr_componentes(photo,{'detections':dets},folder/'ocr.json',folder/'ocr.png')
            saved=json.loads((folder/'ocr.json').read_text(encoding='utf8'))
            self.assertEqual(saved,result);self.assertEqual(saved['raw_token_count'],2)
            self.assertEqual(saved['raw_tokens'][1]['raw_text'],'anotação')
            self.assertEqual(saved['components'][0]['value']['normalized'],'1e-05')
            self.assertEqual(saved['components'][0]['value']['raw_text'],'lOuF')
            self.assertTrue((folder/'ocr.png').is_file())

    def test_tesseract_fallback_runs_when_references_exist_but_values_are_missing(self):
        image=np.full((180,300,3),255,np.uint8);dets=[detection('A','capacitor',20),detection('B','resistor',180)]
        easy=[token('C1',22),token('R1',182)]
        with (patch.object(ocr,'easyocr_disponivel',return_value=True),patch.object(ocr,'_executar_easyocr',return_value=easy),
              patch.object(ocr,'_executar_easyocr_mosaico',return_value=[]),patch.object(ocr,'_executar_tesseract_tsv',return_value=[]) as fallback,
              patch.object(ocr,'_detectar_tokens_por_amostras',return_value=[]),patch.object(ocr,'_aplicar_biblioteca_ocr',side_effect=lambda _,tokens,dets:tokens),
              patch.object(ocr,'_reread_unit_numbers',return_value=([],[]))):
            ocr.extrair_tokens_ocr(image,'tesseract',dets)
        self.assertGreaterEqual(fallback.call_count,1)
