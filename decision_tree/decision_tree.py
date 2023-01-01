# python -m  decision_tree.decision_tree


# ----------- Utilidade
import numpy as np

# ----------- ML
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

# ----------- GRAFICOS
import matplotlib.pyplot as plt

def test_conforme_site():
    # create datasets
    X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)
    # define model
    model = DecisionTreeRegressor()
    # fit model
    model.fit(X, y)
    print('X')
    print(X)
    print('\n-------------------\n')
    print('y')
    print(y)
    # make a prediction
    row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
    yhat = model.predict([row])
    # summarize prediction
    print(yhat[0])


def test_dados_altiok2006():

    
    show_fig_3 = False;
    show_fig_6 = True;

    


    # FEATURES = INPUTS | Ordem: X,P,S
    # a partir do C0 e inputs do time calculo eles ao longo do tempo (só funfa pra bateladas??)
    # time in hour, concs in g/L

    features_names = ['time', 'C_cells_0', 'C_LLA_0', 'C_Lactose_0']
    X = [
            #--------------------
            ### DATA FROM FIG 3
            [0, 1.15, 3.5, 36],
            [1, 1.15, 3.5, 36],
            [2, 1.15, 3.5, 36],
            [3, 1.15, 3.5, 36],
            [4, 1.15, 3.5, 36],
            [5, 1.15, 3.5, 36],
            [6, 1.15, 3.5, 36],
            [7, 1.15, 3.5, 36],
            [8, 1.15, 3.5, 36],
            [9, 1.15, 3.5, 36],
            [10, 1.15, 3.5, 36],
            [11, 1.15, 3.5, 36],

            #--------------------
            ### DATA FROM FIG 6
            [0, 1.5, 8.5, 77.1],
            [1, 1.5, 8.5, 77.1],
            [2, 1.5, 8.5, 77.1],
            [3, 1.5, 8.5, 77.1],
            [4, 1.5, 8.5, 77.1],
            [5, 1.5, 8.5, 77.1],
            [6, 1.5, 8.5, 77.1],
            [7, 1.5, 8.5, 77.1],
            [8, 1.5, 8.5, 77.1],
            [9, 1.5, 8.5, 77.1],
            [10, 1.5, 8.5, 77.1],
            [11, 1.5, 8.5, 77.1],
            [12, 1.5, 8.5, 77.1],
            [13, 1.5, 8.5, 77.1],
            [14, 1.5, 8.5, 77.1],
            [15, 1.5, 8.5, 77.1],
        ]

    # TARGETS = OUTPUTS | Ordem: X, P, S
    outputs_names = ['C_cells_0', 'C_LLA_0', 'C_Lactose_0']
    y = [
            #--------------------
            ### DATA FROM FIG 3
            [1.15, 3.5, 36],
            [1.5, 5, 34],
            [2, 5.8, 34],
            [2.4, 6, 31],
            [2.95 , 7, 25.3],
            [3.6, 9, 21],
            [4.5, 13, 20],
            [5.7, 16.8, 17.3],
            [6.3, 23, 12.5],
            [6.85, 24, 6.8],
            [6.9, 28, 1],
            [6.93, 30, 0.1],

            #--------------------
            ### DATA FROM FIG 6
            [1.5, 8.5, 77.1], #0
            [1.61, 8.55, 70], #1
            [1.8, 9.6, 69.2], #2
            [2.4, 12, 69.2], #3
            [2.9, 12.4, 66 ], #4
            [3.7, 15, 64.5], #5
            [3.9, 19.4, 60], #6
            [4.4, 25, 56], # 7
            [5, 25.9, 46], #8
            [5.3, 28, 40], # 9
            [5.95, 34.7, 35.5], #10
            [6.1, 36, 30.5], #11
            [6.55, 38, 26.2], #12
            [6.8, 40, 23 ], #13
            [6.9, 44, 21], #14
            [6.95, 46, 19.8], #15
    ]

  
    """
    Declara o modelo e faz o fit
    """
    model = DecisionTreeRegressor()
    model.fit(X, y)

    """
    Printa o resultado no tempo 0 (resultado tem que ser idêntico!)
    """
    if(show_fig_3):
        row_for_fig_3 = np.array([
            [0, 1.15, 36, 3.5],
            [1, 1.15, 36, 3.5],
            [3, 1.15, 36, 3.5],
            [3.5, 1.15, 36, 3.5],
            [5, 1.15, 36, 3.5],
            [7, 1.15, 36, 3.5],
            [8, 1.15, 36, 3.5],
            [9, 1.15, 36, 3.5],
            [10, 1.15, 36, 3.5],
            [11, 1.15, 36, 3.5],
            # Esses dois pontos são pra ver como ele extrapola...
            # deu certo, fica estável
            [14, 1.15, 36, 3.5],
            [20, 1.15, 36, 3.5]
            ])
        yhat = model.predict(row_for_fig_3)

        
        plt.plot(row_for_fig_3[:, 0], yhat[:, 0], label='X')
        plt.plot(row_for_fig_3[:, 0], yhat[:, 1], label='P')
        plt.plot(row_for_fig_3[:, 0], yhat[:, 2], label='S')
        # plt.plot(yhat[:, 0], yhat[:, 1], label='first')
        plt.legend()
        plt.show()


    if(show_fig_6):
        # Também deu certo, ficou bem tranquilo!
        row = np.array([
            [0, 1.5, 8.5, 77.1],
            [1, 1.5, 8.5, 77.1],
            [2, 1.5, 8.5, 77.1],
            [3, 1.5, 8.5, 77.1],
            [4, 1.5, 8.5, 77.1],
            [5, 1.5, 8.5, 77.1],
            [6, 1.5, 8.5, 77.1],
            [7, 1.5, 8.5, 77.1],
            [8, 1.5, 8.5, 77.1],
            [9, 1.5, 8.5, 77.1],
            [10, 1.5, 8.5, 77.1],
            [11, 1.5, 8.5, 77.1],
            [12, 1.5, 8.5, 77.1],
            [13, 1.5, 8.5, 77.1],
            [14, 1.5, 8.5, 77.1],
            [15, 1.5, 8.5, 77.1],
            # Extrapolando:
            [17, 1.5, 8.5, 77.1],
            [20, 1.5, 8.5, 77.1],
            ])
        yhat = model.predict(row)

        
        plt.plot(row[:, 0], yhat[:, 0], label='X')
        plt.plot(row[:, 0], yhat[:, 1], label='P')
        plt.plot(row[:, 0], yhat[:, 2], label='S')
        # plt.plot(yhat[:, 0], yhat[:, 1], label='first')
        plt.legend()
        plt.show()
    print("""
    -----------------------
            FINISHED
    -----------------------
    """)
    

def altiok2006_pt2():
    # Será que é melhor eu calcular pelas diferenças? 
    # Tipo a partir do ponto pego o delta e dele é que calculo a concentração
    # faz mais sentido...
    # Porque aí de fato não depende do tempo absoluto (que não existe) mas do tempo que passou
    # e do estado anterior. A mas aí no caso é a mesma coisa que fiz antes,
    # é só ajustar o t_0 e o C0 pra ser o C[t-1] lol. Então dá certo eu acho.
    pass;

if __name__ == '__main__':
    # test_conforme_site();
    
    test_dados_altiok2006();

    # TODO roda isso daí com:
    # lista de conc. células inicial
    # lista conc. LLA incial
    # lista conc. Substrato inicial
    # aí pra cada combinação pega a produção ao fim de até 15 horas
    # calcula a produtividade
    # ordene pela melhor produtividade ou por um fator inventado de performance.

    # TODO agora faz uma interface mais genérica, talvez um objeto
    #
    # cada objeto pode ignorar certos pontos (lista de index ignore data)
    # (os de ruído) pro modelo ficar mais bonitinho
    # 
    # O negócio recebe uma série de objetos
    # cada um tem uma lista de inputs e outputs e um conversor pra transformar eles nas listas,
    # conforme ficou na função aí em cima
    #
    # por fim ele cria e retorna o modelo/solver já pra usar.
    # Isso pode ser usado pra determinar as concentrações ótimas de alimentação
    
    #--------------------------------
    # Alternativamente, pode ser usado para predizer qual tipo de reação? Acho difícil
    # Pq os modelos já deveriam existir e estar carregados...
    # Faz mais sentido Determinar tipos de resina compatíveis com X ou Y substância de interesse
    # e possíveis constantes.
    # Isso eu posso fazer começando pelo artigo da Proteína A?

    
    