import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings


st.set_page_config(page_title="BPJS Fraud Case Detection", page_icon="⛳️", layout='centered', initial_sidebar_state="collapsed"
                   )

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> 🚨 BPJS Fraud Case Detection</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 2])

    with col1:
        with st.expander(" ℹ️ Information", expanded=True):
            st.write("""
            BPJS fraud detection is a pivotal aspect in ensuring the integrity and effectiveness of the health insurance system. Identifying fraudulent activities within the BPJS framework is crucial to maintain trust and financial stability. Various methodologies and algorithms are employed to detect suspicious patterns and behaviors. However, the detection systems can vary in their accuracy and efficiency.
                     
            In the realm of health insurance, it's imperative that fraud detection mechanisms are both robust and precise. Inaccuracies or oversights in identifying fraudulent claims can lead to substantial financial losses and undermine the trust of policyholders.
            
            
            🧬 Looking for disease diagnosis codes? Find them [here](https://www.icd10data.com/ICD10CM/Codes)!
                     
            ---
            *Copyright © 2024 by DaMi-2324-Team05*
        
            *Made with  ❤️  by DaMi-2324-Team05*
                     
            *(Irma, Marcel, Dani, Jevania)*
                             
            """)
        '''
        ## How does it work ❓ 
        Complete all the parameters, and the machine learning model will predict whether a BPJS claim is fraud or not fraud based on various factors.
        '''

    with col2:
        st.subheader(
            "👩‍⚕️ Find out the fraud occurance in BPJS Visitation")
        
        KDKC = st.number_input("BPJS Health Care Office branch area code", 1, 2606)
        DATI2 = st.number_input("District/city code", 1, 528)
		
        typeppk_list = ['B', 'C', 'D', 'GD', 'HD', 'I1', 'I2', 'I3', 'I4', 'KB', 'KC',
                        'KG', 'KI', 'KJ', 'KL', 'KM', 'KO', 'KP', 'KT', 'KU', 'SA', 'SB', 'SC', 'SD']


        TYPEPPK = st.selectbox("Hospital type code", typeppk_list)
        typeppk_dict = dict()

        for typeppk in typeppk_list:
            typeppk_dict["typeppk_" + typeppk] = 0

        typeppk_dict["typeppk_" + TYPEPPK] = 1
		
        jkpst_list = ["L", "P"]
        JKPST = st.selectbox("Select gender: 0 -> Male || 1 -> Female", jkpst_list)
        JKPST = 1 if JKPST == "P" else 0
    
        UMUR = st.number_input("JKN-KIS participant age at the time of hospital service", 0, 109)
        JNSPELSEP = st.number_input("Level of service: 1 -> inpatient || 2 -> outpatient", 1, 2)
        LOS = st.number_input("Length of time the JKN-KIS participant was hospitalized", 0, 592)

        cm_codes = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                    'L', 'M', 'N', 'O', 'P', 'Q', 'S', 'T', 'U', 'V', 'W', 'Z']
        CMG = st.selectbox("CMG (Case Mix Group) classification", cm_codes)
        cmg_dict = dict()
        for cmg in cm_codes:
            cmg_dict["cmg_" + cmg] = 0

        cmg_dict["cmg_" + CMG] = 1

        SEVERITYLEVEL = st.number_input("Level of urgency", 0, 3)

        diagnosis_options = [
            'c00_d48', 'd50_d89', 'e00_e90', 'f00_f99', 'g00_g99', 'h00_h59', 'h60_h95', 'i00_i99', 'j00_j99', 'k00_k93', 'l00_l99', 'm00_m99', 'n00_n99', 'o00_o99', 'p00_p96', 'q00_q99', 'r00_r99', 's00_t98', 'u00_u85', 'z00_z99'
             ]
        DIAGPRIMER = st.selectbox("Primary diagnosis code", diagnosis_options)
        diag_dict = dict()
        for diag in diagnosis_options:
            diag_dict["diagprimer_" + diag] = 0

        diag_dict["diagprimer_" + DIAGPRIMER] = 1


        dx2_options = [
             'dx2_a00_b99', 'dx2_c00_d48', 'dx2_d50_d89', 'dx2_e00_e90', 'dx2_f00_f99', 'dx2_g00_g99', 'dx2_h00_h59', 'dx2_h60_h95', 'dx2_i00_i99', 'dx2_j00_j99', 'dx2_l00_l99', 'dx2_m00_m99', 'dx2_n00_n99', 'dx2_o00_o99', 'dx2_p00_p96', 'dx2_q00_q99', 'dx2_r00_r99', 'dx2_s00_t98', 'dx2_v01_y98', 'dx2_z00_z99'
		]
        DX2 = st.selectbox("Secondary diagnosis code", dx2_options)
        dx2_dict = dict()
        for dx2 in dx2_options:
            dx2_dict[dx2] = 0

        dx2_dict[DX2] = 1


        proc_options = [
             'proc00_13', 'proc14_23', 'proc24_27', 'proc28_28', 'proc29_31', 'proc_32_38', 'proc39_45', 'proc46_51', 'proc52_57', 'proc58_62', 'proc63_67', 'proc68_70', 'proc71_73', 'proc74_75', 'proc76_77', 'proc78_79', 'proc80_99', 'proce00_e99'
             ]

        PROC = st.selectbox("Procedure code", proc_options)
        proc_dict = dict()
        for proc in proc_options:
            proc_dict[proc] = 0

        proc_dict[PROC] = 1

                                     
        feature_list = list()
        for d in [typeppk_dict, cmg_dict, diag_dict]:
            feature_list.extend(d.values())

        feature_list.extend((KDKC, DATI2))
        feature_list.extend((JKPST, UMUR, JNSPELSEP, LOS, SEVERITYLEVEL))
        feature_list.extend(dx2_dict.values())
        feature_list.extend(proc_dict.values())

        single_pred = np.array(feature_list).reshape(1, -1)

        if st.button('Predict'):
            #  print(single_pred)
             loaded_model = load_model('model.pkl')
             prediction = loaded_model.predict(single_pred).item()
             prediction = np.round(prediction).astype(int)
             col1.write('''
                        ## Results
                        ''')
             prediction_result = 'Fraud' if prediction == 1 else 'Not Fraud'
             col1.success(f"{prediction_result} is the prediction result for your BPJS claim")
      # code for html

    st.warning(
    	"Note: This A.I application is for educational/demo purposes only and cannot be relied upon. Check the source code [here](https://github.com/marceljsh/DaMi-FraudDetection-BPJS-CNN)")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
<style>
#MainMenu {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
