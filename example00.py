import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal

st.title('Graph Tool')
st.text('Python Version')

st.sidebar.title('Function Selection')
fun = st.sidebar.selectbox('Options', ['a*sin(bx)', 'ax^n', 'mx+b', 'e^at', 'Pulse','Triangle','Square','Pulse Train','delta','X1','X2','X3','X4'])
x = np.arange(0, 2*np.pi, 0.1)

if fun == 'a*sin(bx)':
    st.write('Sine Function')
    a = st.sidebar.slider('Select a:', 1.0, 15.0, 9.0, 0.5)
    b = st.sidebar.number_input('Select b:', 1, 15)
    y = a*np.sin(b*x)
elif fun == 'ax^n':
    st.text('Poly Function')
    a = st.sidebar.slider('a:', 1, 15)
    n = st.sidebar.number_input('n:', 1, 15)
    y = a*x**n
elif fun == 'mx+b':
    st.text('Line Function')
    m = st.sidebar.slider('m:', 1, 15)
    b = st.sidebar.number_input('b:', 1, 15)
    y = m*x+b
elif fun == 'e^at':
    st.text('Euler Function')
    a = st.sidebar.number_input('a:', 1, 15)
    y = np.exp(a*x)
elif fun == 'Pulse':
    st.text('Pulse Function')
    a = st.sidebar.number_input('amplitude:', 1, 15)
    w = st.sidebar.number_input('width:', 1, 15)
    x = np.arange(-w, w, 0.1)
    y = np.zeros(len(x))
    y[(-w/2 < x) & (x < w/2)] = a
elif fun == 'Triangle':
    st.text('Triangle Function')
    a = st.sidebar.number_input('amplitude:', 1, 15)
    w = st.sidebar.number_input('Frecuency:', 1, 15)
    x = np.arange(-5, 5, 0.1)
    y = a*signal.sawtooth(2 * np.pi * w * x)
elif fun == 'Square':
    st.text('Triangle Function')
    a = st.sidebar.number_input('amplitude:', 1, 15)
    w = st.sidebar.number_input('Frecuency:', 1, 15)
    x = np.arange(-5, 5, 0.1)
    y = a*signal.square(2 * np.pi * w * x)
elif fun == 'Pulse Train':
    st.text('PT Function')
    x = st.sidebar.text_input('x []:')
    y = st.sidebar.text_input('y []:')
    x = np.array(np.matrix(x)).ravel()
    y = np.array(np.matrix(y)).ravel()
elif fun == 'X1':
    st.text('Sustentación')
    x = np.arange(-20, 20, 0.1)
    f1 = np.zeros(len(x))
    f1[(-5 < x)] = 1
    f2 = np.zeros(len(x))
    f2[(3 < x)] = 1
    y = (f1 - f2)*np.sin(0.1*np.pi*x)
elif fun == 'X2':
    st.text('Sustentación')
    x = np.arange(-20, 20, 0.1)
    f1 = np.zeros(len(x))
    f1[(6 < x)] = 1
    f2 = np.zeros(len(x))
    f2[(-12 < x)] = 1
    y = (f1 + f2)*(3.1**x)
elif fun == 'X3':
    st.text('Sustentación')
    x = np.arange(-20, 20, 0.1)
    f1 = np.zeros(len(x))
    f1[(-3 < x)] = 1
    f2 = np.zeros(len(x))
    f2[(-8 < x)] = 1
    y = (f1 + f2)*np.exp(4*x)
elif fun == 'X4':
    st.text('Sustentación')
    x = np.arange(-20, 20, 0.1)
    f1 = np.zeros(len(x))
    f1[(-4 < x)] = 1
    f2 = np.zeros(len(x))
    f2[(2 < x)] = 1
    y = (f1 - f2)*np.cos(0.5*np.pi*x)
    
dis = st.sidebar.checkbox('Discrete')
    
plt.figure(figsize=(10,5))
if dis:
    plt.stem(x,y)
else:
    plt.plot(x,y) 
st.pyplot(clear_figure=True)  

st.sidebar.title('Operation Selection')
op = st.sidebar.selectbox('Operations Available:', ['Time Scaling', 'Time Shifting', 'Amplitude Scaling'])

if op == 'Time Scaling':
    A = st.sidebar.number_input('Select A, where g(x)= y(Ax):', 0.0, 15.0,1.0)
    st.write('Time Scaled by: ', A)
    z = y
elif op == 'Time Shifting':
    A = st.sidebar.number_input('Select A, where g(x)= y(x+A):', -15, 15,0)
    st.write('Time Shifted by: ', A)
    z = y
elif op == 'Amplitude Scaling':
    A = st.sidebar.number_input('Select A, where g(x)= A*y(x):', -15, 15,1)
    st.write('Scaled by: ', A)
    z = A*y

animation = st.sidebar.button('Animate')
Graph = st.empty()

if animation:
    frames = 10
    if op == 'Time Scaling':
        for i in range(frames+1):
            plt.plot(x,y, label= 'Original')
            plt.plot(x/(1+ i*(A-1)/frames),y, label= 'Time Scaled')
            plt.legend(loc="upper left")
            plt.xlabel('x(t)')
            plt.ylabel('y(t)')
            plt.xlim(0,2*np.pi+np.pi/A)
            Graph.pyplot()
    elif op == 'Time Shifting':
        for i in range(frames+1):
            plt.plot(x,y, label= 'Original')
            plt.plot(x - i*(A)/frames,y,label= 'Time Shifted')
            plt.legend(loc="upper left")
            plt.xlabel('x(t)')
            plt.ylabel('y(t)')
            plt.xlim(0,max(x))
            Graph.pyplot()
    elif op == 'Amplitude Scaling':
        for i in range(frames+1):
            plt.plot(x,y, label= 'Original')
            plt.plot(x,y*(1 + i*(A-1)/frames), label= 'Amplitude Scaled')
            plt.legend(loc="upper left")
            plt.xlabel('x(t)')
            plt.ylabel('y(t)')
            plt.ylim(min(z)-1, max(z)+1)
            plt.xlim(0, max(x)+1)
            Graph.pyplot()
    



        

