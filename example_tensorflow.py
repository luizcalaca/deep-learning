#comente
import numpy as np
import tensorflow as tf

#Um objeto de Sessão encapsula o ambiente em que os objetos de Operação são executados e os objetos do Tensor são avaliados
sess = tf.Session()

#Função que imprime o tipo e valor da variável x
def print_tf(x):
    print("TIPO: \n %s" % (type(x)))
    print("Valor: \n %s" % (x))
hello = tf.constant("www.deeplearningbrasil.com.br")
print_tf(hello)

#Executando a operação hello
hello_out = sess.run(hello)
print_tf(hello_out)

#criação de dois (nós) tensors constantes
a = tf.constant(1.5)
b = tf.constant(2.5)
print_tf(a)
print_tf(b)

#Executando os tensors 'a' e 'b'
a_out = sess.run(a)
b_out = sess.run(b)
print_tf(a_out)
print_tf(b_out)

#Adição do tensor 'a' com o 'b'
a_plus_b = tf.add(a, b)
print_tf(a_plus_b)

#Inicializando a sessão e rodando a_plus_b dando como resultado a soma dos valores contidos nos tensors 'a' e 'b'
a_plus_b_out = sess.run(a_plus_b)
print_tf(a_plus_b_out)

#Rodando a multiplicação dos valores do tensor 'a' por 'b'
a_mul_b = tf.multiply(a, b)
a_mul_b_out = sess.run(a_mul_b)
print_tf(a_mul_b_out)

#Criação e inicialização de todas as variáveis globais
weight = tf.Variable(tf.random_normal([5, 2], stddev=0.1))
init = tf.global_variables_initializer()
sess.run(init)
print_tf(weight)

#Havia ocorrido um erro aqui pela variável 'weight' não ter sido inicializada
weight_out = sess.run(weight)
print_tf(weight_out)

#Rodando o weight que é uma variável com valor randômico
weight_out = sess.run(weight)
print_tf(weight_out)
print ("INITIALIZING ALL VARIALBES")

#O placeholder insere um espaço reservado para um tensor que sempre será alimentado
x = tf.placeholder(tf.float32, [None, 5])
print_tf(x)

#O 'matmul' retornará um tensor através do produto de outros dois tensors
oper = tf.matmul(x, weight)
print_tf(oper)

#Criando um array de números randômicos 'data' pelo Numpy, e, rodando o 'oper' para a multiplicação de dois tensors
data = np.random.rand(1, 5)
oper_out = sess.run(oper, feed_dict={x: data})
print_tf(oper_out)

#Criando um array de números randômicos 'data' pelo Numpy, e, rodando o 'oper' para a multiplicação de dois tensors
data = np.random.rand(2, 5)
oper_out = sess.run(oper, feed_dict={x: data})
print_tf(oper_out)
