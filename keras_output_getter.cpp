#include "keras_output_getter.h"
#include <QProcess>
#include <QDebug>

Keras_output_getter::Keras_output_getter(QString path_to_script, QObject *parent) : QObject(parent)
{
    pr = new QProcess;
    connect(pr,SIGNAL(errorOccurred(QProcess::ProcessError)),this,SLOT(handleErrors(QProcess::ProcessError)));
    //connect(pr,SIGNAL(readyReadStandardOutput()),this,SLOT(readModelOutout()));
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("PYTHONPATH", "/home/mike2/PycharmProjects/pythonProject"); // Add an environment variable
    pr->setProcessEnvironment(env);

    m_py_script = path_to_script;
}


void Keras_output_getter::handleErrors(QProcess::ProcessError err)
{
    qDebug() << "Err Occured2";
}

void Keras_output_getter::readModelOutout()
{
     QString p_stdout = pr->readAll();
     qDebug() << "ff" << p_stdout;

}

void Keras_output_getter::run()
{
    pr->start("/usr/bin/python3", QStringList() <<  m_py_script);
    qDebug() << "!!!" << m_py_script;
}
