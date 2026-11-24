import unittest
import cv2
import numpy as np

from backend.terminal_detection import detectar_terminais_componentes


def bbox(x1,y1,x2,y2):
    return {"x1":x1,"y1":y1,"x2":x2,"y2":y2,"width":x2-x1,"height":y2-y1}


def draw_bjt(kind: str, direction: str = "outward"):
    img=np.full((220,220,3),245,np.uint8)
    cv2.line(img,(95,65),(95,155),(20,20,20),4)
    cv2.line(img,(40,110),(95,110),(20,20,20),3)
    upper=((95,90),(155,55)); lower=((95,130),(155,165))
    cv2.line(img,*upper,(20,20,20),3)
    cv2.line(img,*lower,(20,20,20),3)
    br=lower
    (ax,ay),(bx,by)=br
    if direction=="outward":
        tip_t, tail_t=.74,.54
    else:
        tip_t, tail_t=.38,.60
    tip=(int(ax+(bx-ax)*tip_t),int(ay+(by-ay)*tip_t))
    cx=ax+(bx-ax)*tail_t; cy=ay+(by-ay)*tail_t
    dx,dy=bx-ax,by-ay; norm=(dx*dx+dy*dy)**.5
    px,py=-dy/norm,dx/norm
    p1=(int(cx+px*9),int(cy+py*9)); p2=(int(cx-px*9),int(cy-py*9))
    cv2.line(img,p1,tip,(20,20,20),2); cv2.line(img,p2,tip,(20,20,20),2)
    info={"schema_version":"strict-test","components":[{
        "component_id":"Q1","type":kind,"bbox":bbox(55,45,170,180)
    }]}
    return img,info


class StrictGeometryTests(unittest.TestCase):
    def test_npn_class_is_used_before_emitter_assignment(self):
        img,info=draw_bjt("transistor_npn","outward")
        out=detectar_terminais_componentes(info,220,220,img)
        comp=out["components"][0]
        self.assertEqual(comp["transistor_geometry"].get("transistor_type"),"NPN")
        self.assertEqual(comp["transistor_geometry"].get("expected_arrow_direction"),"outward")
        self.assertEqual(comp["transistor_geometry"].get("arrow_branch"),"lower")
        pins={p["pin"]:p for p in comp["terminals"]}
        self.assertGreater(pins["E"]["position"]["y"],pins["C"]["position"]["y"])
        self.assertFalse(pins["E"].get("identity_uncertain",False))

    def test_pnp_class_requires_inward_arrow(self):
        img,info=draw_bjt("transistor_pnp","inward")
        out=detectar_terminais_componentes(info,220,220,img)
        comp=out["components"][0]
        self.assertEqual(comp["transistor_geometry"].get("transistor_type"),"PNP")
        self.assertEqual(comp["transistor_geometry"].get("expected_arrow_direction"),"inward")
        self.assertEqual(comp["transistor_geometry"].get("arrow_branch"),"lower")
        pins={p["pin"]:p for p in comp["terminals"]}
        if pins:
            self.assertGreater(pins["E"]["position"]["y"],pins["C"]["position"]["y"])
        else:
            # Política segura: se a direção da seta não puder ser provada,
            # não publica B/C/E potencialmente errados.
            self.assertTrue(comp["terminal_detection"]["review_required"])
            self.assertTrue(comp["transistor_geometry"].get("training_required"))

    def test_capacitor_terminals_follow_center_of_parallel_plates(self):
        img=np.full((210,210,3),245,np.uint8)
        cv2.line(img,(65,88),(145,88),(20,20,20),3)
        cv2.line(img,(65,106),(145,106),(20,20,20),3)
        cv2.line(img,(105,35),(105,88),(20,20,20),3)
        cv2.line(img,(105,106),(105,170),(20,20,20),3)
        info={"schema_version":"strict-test","components":[{
            "component_id":"C1","type":"capacitor","bbox":bbox(45,30,165,175)
        }]}
        out=detectar_terminais_componentes(info,210,210,img)
        comp=out["components"][0]
        self.assertEqual(comp["orientation"]["source"],"capacitor_parallel_plates")
        pins=comp["terminals"]
        self.assertAlmostEqual(pins[0]["position"]["x"],105,delta=5)
        self.assertAlmostEqual(pins[1]["position"]["x"],105,delta=5)
        self.assertLess(pins[0]["position"]["y"],88)
        self.assertGreater(pins[1]["position"]["y"],106)

    def test_vertical_rectangular_resistor_uses_body_not_bbox(self):
        img=np.full((210,230,3),245,np.uint8)
        cv2.rectangle(img,(102,70),(124,140),(20,20,20),3)
        cv2.line(img,(113,30),(113,70),(20,20,20),3)
        cv2.line(img,(113,140),(113,180),(20,20,20),3)
        info={"schema_version":"strict-test","components":[{
            "component_id":"R1","type":"resistor","bbox":bbox(55,25,175,185)
        }]}
        out=detectar_terminais_componentes(info,230,210,img)
        comp=out["components"][0]
        self.assertEqual(comp["orientation"]["axis"],"vertical")
        self.assertEqual(comp["orientation"]["source"],"resistor_body_rectangle")
        for pin in comp["terminals"]:
            self.assertAlmostEqual(pin["position"]["x"],113,delta=6)

if __name__ == '__main__':
    unittest.main()
