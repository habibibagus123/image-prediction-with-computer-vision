import streamlit as st
import eda
import prediction


page = st.sidebar.selectbox(label='Select Page:', options=['Home Page', 'Exploration Data Analysis', 'Model Prediksi'])

if page == 'Home Page':
    st.header('Welcome Page') 
    st.write('')
    st.write('Graded Challenge 7')
    st.write('Nama      : Habibi Bagu Suliano')
    st.write('Batch     : HCK-009')
    st.write('Tujuan    : program ini dibuat untuk memprediksi gambar')
    st.write('')
    st.caption('Silahkan pilih menu lain di Select Box pada sebelah kiri layar anda untuk memulai!')
    st.write('')
    st.write('')
    with st.expander("Latar Belakang"):
        st.caption('menentukan prediksi gambar untuk memenuhi objective graded challange 7 ini oleh karena itu saya membuat prediksi gambar dengan 6 class.')

    with st.expander("Problem Statement"):
            st.caption('memprediksi gambar')

    with st.expander("Kesimpulan"):
        st.caption('kesimpulan yang didapat adalah dengan model cnn didappatkan keakuratan yaitu hanya 66% oleh karena itu saya melakukan improvment model dengan tranfer learning menggunakan mobilenet dan didapatkan imprvement keakuratan menjadi 82%')
elif page == 'Exploration Data Analysis':
    eda.run()
else:
     prediction .run()