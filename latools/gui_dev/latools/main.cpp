#include "latools.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    latools w;
    w.show();

    return a.exec();
}
