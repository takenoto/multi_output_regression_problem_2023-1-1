
# python -m linear_regression.my_test

import matplotlib.pyplot as plt

def test_de_forma_com_prints():
    # Isso aqui sou eu caducando pra conseguir entender
    # como é o input e preparar o meu


    # linear regression for multioutput regression
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    # create datasets
    # features são os inputs
    # targets são os outputs
    X, y = make_regression(n_samples=7, n_features=3, n_informative=5, n_targets=2, random_state=1, noise=0.5)
    print('X = ')
    print(f'{X}')
    print('--------------------------------')
    print('\n\n\n')
    print('y = ')
    print(f'{y}')
    print('--------------------------------')
    # define model
    model = LinearRegression()
    # fit model
    model.fit(X, y)
    # make a prediction
    # row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
    row = [1, 2, 3.3]
    yhat = model.predict([row])
    # summarize prediction
    print(yhat[0])

# TODO agora faz o mesmo igualzinho, só que com seus dados do artigo
def test_artigo_glicerina_aracanjo():
    """
    Esse daqui é realmente só pra eu ver se consegue prever as coisas
    de forma minimamente viável, e já testo 2 inputs
    """
    # linear regression for multioutput regression
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    # create datasets
    # features são os inputs
    # targets são os outputs
    # X, y = make_regression(n_samples=7, n_features=3, n_informative=5, n_targets=2, random_state=1, noise=0.5)
    
    """
    Objetivo: descobrir a conc. de saída de LA em cada etapa para uma dada temperatura
    Columns = dados que vão ser entrados, em ordem pra eu saber quem é quem
    
    ***Input: 
        - Conc_La_feed
        - Temperature
    
    ***Output:
        - conc_LA_elution
        - 
    """
    # FEATURES = INPUT
    # PARA RESINA AMBERLITE IRA 67
    # Temp kelvin, conc g/L, Q é vazão em mL/min
    # Dados tables 4, 6
    features = {
        'columns_names': ['temperature, conc_LA_feed', 'conc_glycerol_feed', 'Q'],
        'values': [
            [30+273, 345, 230, 2.5],
            [30+273, 138, 34, 2.5]
        ]
    }

    X = features['values'];    

    # TARGETS = OUTPUT
    targets = {
        'columns_names':['conc_LA_adsorption', 'conc_glycerol_adsorption'],
        'values': [
            [154, 131],
            [68, 17.55]
        ],
    }

    y = targets['values'];

    # define model
    model = LinearRegression()
    # fit model
    model.fit(X, y)
    # make a prediction
    # Exatamente o primeiro e o segundo pontos de dado
    for i in range(2):
        print(f'i = {i}')
        print('calculando:')
        row = features['values'][i]
        yhat = model.predict([row])
        print('valores obtidos:')
        print(yhat[0])
        print(targets['values'][i])
        print('---------------------------\n')

    # Agora itera entre os 2 pontos variando apenas a conc LA
    max_number = 30;
    conc_feed_la_list = []
    # conc_glyc_list = []
    conc_adsorption_la = []
    conc_adsorption_glyc = []
    for i in range (max_number):
        C_LA_feed = (i/max_number)*(345-138) + 138
        C_glyc_feed = 34
        row = [30+273, C_LA_feed, C_glyc_feed, 2.5]
        yhat = model.predict([row])
        result = yhat[0]
        conc_feed_la_list.append(C_LA_feed)
        conc_adsorption_la.append(result[0])
        conc_adsorption_glyc.append(result[1])

    
    plt.plot(conc_feed_la_list, conc_adsorption_la, label='la adsorption')
    plt.plot(conc_feed_la_list, conc_adsorption_glyc, label='glyc adsorption')
    plt.legend()
    plt.show()

    pass;



if __name__ == '__main__':
    # test_de_forma_com_prints()

    # Resultados absolutamente satisfatórios, uma pena serem poucos dados
    # test_artigo_glicerina_aracanjo(),

    # o que eu quero DEFINITIVAMENTE NÃO É UMA REGRESSÃO LINEAR!
    pass;