#include "model_driving.h"
#include <iostream>
#include <capnp/c++.capnp.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include "log.capnp.h"
#include "main.h"
#include "modelfunc.h"
#include <math.h>
#include <QProcess>
#include <QDebug>
#include "keras_output_getter.h"
#include <QGuiApplication>
#include <QProcess>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QMap>
#include <QPair>
#include "/usr/include/eigen3/Eigen/Dense"

#include "messaging.hpp"
#include "impl_zmq.hpp"
#include <QFile>
#include <capnp/serialize-packed.h>
#include <typeinfo>


#define PATH_IDX 0
#define LL_IDX PATH_IDX + MODEL_PATH_DISTANCE*2 + 1
#define RL_IDX LL_IDX + MODEL_PATH_DISTANCE*2 + 2
#define LEAD_IDX RL_IDX + MODEL_PATH_DISTANCE*2 + 2
#define LONG_X_IDX LEAD_IDX + MDN_GROUP_SIZE*LEAD_MDN_N + SELECTION
#define LONG_V_IDX LONG_X_IDX + TIME_DISTANCE*2
#define LONG_A_IDX LONG_V_IDX + TIME_DISTANCE*2
#define DESIRE_STATE_IDX LONG_A_IDX + TIME_DISTANCE*2
#define META_IDX DESIRE_STATE_IDX + DESIRE_LEN
#define POSE_IDX META_IDX + OTHER_META_SIZE + DESIRE_PRED_SIZE
#define OUTPUT_SIZE  POSE_IDX + POSE_SIZE

Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE - 1> vander;

Model_driving::Model_driving(QProcess* readKerasProcceess,QObject *parent) : QObject(parent)
{
    m_readKerasProcess = readKerasProcceess;
    ModelDataRaw raw_model;
    outputIdx2memberStruct = new QMap<int, float*>;

    Context *msg_context = Context::create();
    Model_driving::model_sock = PubSocket::create(msg_context, "model");

}

void Model_driving::fillRawModelStructMembers(QJsonArray json, int idx)
{
    float* tmp = new float[json.at(idx).toArray().count()];
    for (int i = 0; i < json.at(idx).toArray().count(); i++)
     {
         tmp[i] = static_cast<float>(json.at(idx).toArray().at(i).toDouble());
     }
    outputIdx2memberStruct->insert(idx,tmp);
}

void Model_driving::formRawModel()
{
    //path, ll, rl, lead, long_x, long_v, long_a, desire_state, meta, desire_pred, pose

    QString outputJSON = m_readKerasProcess->readAllStandardOutput().replace("\n","").trimmed();
    QJsonDocument doc = QJsonDocument::fromJson(outputJSON.toUtf8());
    if(!doc.isNull())
    {
        QJsonDocument doc = QJsonDocument::fromJson(outputJSON.toUtf8());
        QJsonArray json = doc.array();
        for (int m = 0; m < json.count(); m++)
        {
            fillRawModelStructMembers(json,m);
        }
        raw_model.path = outputIdx2memberStruct->value(0);
        raw_model.left_lane = outputIdx2memberStruct->value(1);
        raw_model.right_lane = outputIdx2memberStruct->value(2);
        raw_model.lead = outputIdx2memberStruct->value(3);
        raw_model.long_x = outputIdx2memberStruct->value(4);
        raw_model.long_v = outputIdx2memberStruct->value(5);
        raw_model.long_a = outputIdx2memberStruct->value(6);
        raw_model.desire_state = outputIdx2memberStruct->value(7);
        raw_model.meta = outputIdx2memberStruct->value(8);
        raw_model.pose = outputIdx2memberStruct->value(10);

        publish_model(raw_model);


    }

}

void Model_driving::poly_fit(float *in_pts, float *in_stds, float *out) {
  // References to inputs
  Eigen::Map<Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> > pts(in_pts, MODEL_PATH_DISTANCE);
  Eigen::Map<Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> > std(in_stds, MODEL_PATH_DISTANCE);
  Eigen::Map<Eigen::Matrix<float, POLYFIT_DEGREE - 1, 1> > p(out, POLYFIT_DEGREE - 1);

  float y0 = pts[0];
  pts = pts.array() - y0;

  // Build Least Squares equations
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, POLYFIT_DEGREE - 1> lhs = vander.array().colwise() / std.array();
  Eigen::Matrix<float, MODEL_PATH_DISTANCE, 1> rhs = pts.array() / std.array();

  // Improve numerical stability
  Eigen::Matrix<float, POLYFIT_DEGREE - 1, 1> scale = 1. / (lhs.array()*lhs.array()).sqrt().colwise().sum();
  lhs = lhs * scale.asDiagonal();

  // Solve inplace
  Eigen::ColPivHouseholderQR<Eigen::Ref<Eigen::MatrixXf> > qr(lhs);
  p = qr.solve(rhs);

  // Apply scale to output
  p = p.transpose() * scale.asDiagonal();
  out[3] = y0;
}

void Model_driving::fill_path(cereal::ModelData::PathData::Builder path, const float * data, bool has_prob, const float offset) {

  //qDebug() <<"DATA"<<data[0];
  float points_arr[MODEL_PATH_DISTANCE];
  float stds_arr[MODEL_PATH_DISTANCE];
  float poly_arr[POLYFIT_DEGREE];
  float std;
  float prob;
  float valid_len;

  valid_len =  data[MODEL_PATH_DISTANCE*2];
  for (int i=0; i<MODEL_PATH_DISTANCE; i++)
  {
    points_arr[i] = data[i] + offset;
    // Always do at least 5 points
    if (i < 5 || i < valid_len) {
      stds_arr[i] = softplus(data[MODEL_PATH_DISTANCE + i]);
    } else {
      stds_arr[i] = 1.0e3;
    }
  }
  if (has_prob) {
    prob =  sigmoid(data[MODEL_PATH_DISTANCE*2 + 1]);
  } else {
    prob = 1.0;
  }
  std = softplus(data[MODEL_PATH_DISTANCE]);
  poly_fit(points_arr, stds_arr, poly_arr);

  //if (std::getenv("DEBUG")){
    kj::ArrayPtr<const float> stds(&stds_arr[0], ARRAYSIZE(stds_arr));
    path.setStds(stds);

    kj::ArrayPtr<const float> points(&points_arr[0], ARRAYSIZE(points_arr));
    path.setPoints(points);
  //}

  kj::ArrayPtr<const float> poly(&poly_arr[0], ARRAYSIZE(poly_arr));
  path.setPoly(poly);
  path.setProb(prob);
  path.setStd(std);
}

void Model_driving::fill_lead(cereal::ModelData::LeadData::Builder lead, const float * data, int mdn_max_idx, int t_offset) {
  const double x_scale = 10.0;
  const double y_scale = 10.0;

  lead.setProb(sigmoid(data[LEAD_MDN_N*MDN_GROUP_SIZE + t_offset]));
  lead.setDist(x_scale * data[mdn_max_idx*MDN_GROUP_SIZE]);
  lead.setStd(x_scale * softplus(data[mdn_max_idx*MDN_GROUP_SIZE + MDN_VALS]));
  lead.setRelY(y_scale * data[mdn_max_idx*MDN_GROUP_SIZE + 1]);
  lead.setRelYStd(y_scale * softplus(data[mdn_max_idx*MDN_GROUP_SIZE + MDN_VALS + 1]));
  lead.setRelVel(data[mdn_max_idx*MDN_GROUP_SIZE + 2]);
  lead.setRelVelStd(softplus(data[mdn_max_idx*MDN_GROUP_SIZE + MDN_VALS + 2]));
  lead.setRelA(data[mdn_max_idx*MDN_GROUP_SIZE + 3]);
  lead.setRelAStd(softplus(data[mdn_max_idx*MDN_GROUP_SIZE + MDN_VALS + 3]));
}

void Model_driving::fill_meta(cereal::ModelData::MetaData::Builder meta, const float * meta_data) {
  kj::ArrayPtr<const float> desire_state(&meta_data[0], DESIRE_LEN);
  meta.setDesireState(desire_state);
  meta.setEngagedProb(meta_data[DESIRE_LEN]);
  meta.setGasDisengageProb(meta_data[DESIRE_LEN + 1]);
  meta.setBrakeDisengageProb(meta_data[DESIRE_LEN + 2]);
  meta.setSteerOverrideProb(meta_data[DESIRE_LEN + 3]);
  kj::ArrayPtr<const float> desire_pred(&meta_data[DESIRE_LEN + OTHER_META_SIZE], DESIRE_PRED_SIZE);
  meta.setDesirePrediction(desire_pred);
}

void Model_driving::fill_longi(cereal::ModelData::LongitudinalData::Builder longi, const float * long_v_data, const float * long_a_data) {
  // just doing 10 vals, 1 every sec for now
  float speed_arr[TIME_DISTANCE/10];
  float accel_arr[TIME_DISTANCE/10];
  for (int i=0; i<TIME_DISTANCE/10; i++) {
    speed_arr[i] = long_v_data[i*10];
    accel_arr[i] = long_a_data[i*10];
  }
  kj::ArrayPtr<const float> speed(&speed_arr[0], ARRAYSIZE(speed_arr));
  longi.setSpeeds(speed);
  kj::ArrayPtr<const float> accel(&accel_arr[0], ARRAYSIZE(accel_arr));
  longi.setAccelerations(accel);
}


void Model_driving::publish_model(ModelDataRaw net_outputs)
{
    qDebug() << net_outputs.path;
    capnp::MallocMessageBuilder msg;
        cereal::Event::Builder event = msg.initRoot<cereal::Event>();
        //event.setLogMonoTime(nanos_since_boot());

        auto framed = event.initModel();
        //framed.setFrameId(frame_id);
        //framed.setTimestampEof(timestamp_eof);

        auto lpath = framed.initPath();
        fill_path(lpath, net_outputs.path, false, 0);
        auto left_lane = framed.initLeftLane();
        fill_path(left_lane, net_outputs.left_lane, true, 1.8);
        auto right_lane = framed.initRightLane();
        fill_path(right_lane, net_outputs.right_lane, true, -1.8);
        auto longi = framed.initLongitudinal();
        fill_longi(longi, net_outputs.long_v, net_outputs.long_a);


        // Find the distribution that corresponds to the current lead
        int mdn_max_idx = 0;
        int t_offset = 0;
        for (int i=1; i<LEAD_MDN_N; i++) {
          if (net_outputs.lead[i*MDN_GROUP_SIZE + 8 + t_offset] > net_outputs.lead[mdn_max_idx*MDN_GROUP_SIZE + 8 + t_offset]) {
            mdn_max_idx = i;
          }
        }
        auto lead = framed.initLead();
        fill_lead(lead, net_outputs.lead, mdn_max_idx, t_offset);
        // Find the distribution that corresponds to the lead in 2s
        mdn_max_idx = 0;
        t_offset = 1;
        for (int i=1; i<LEAD_MDN_N; i++) {
          if (net_outputs.lead[i*MDN_GROUP_SIZE + 8 + t_offset] > net_outputs.lead[mdn_max_idx*MDN_GROUP_SIZE + 8 + t_offset]) {
            mdn_max_idx = i;
          }
        }
        auto lead_future = framed.initLeadFuture();
        fill_lead(lead_future, net_outputs.lead, mdn_max_idx, t_offset);


        auto meta = framed.initMeta();
        fill_meta(meta, net_outputs.meta);


        // send message
        QFile myFile("/home/mike/emelya/dummy.txt");
        myFile.open(QIODevice::ReadWrite);
        int fd = myFile.handle();
          //writePackedMessageToFd(fd, msg);

        auto words = capnp::messageToFlatArray(msg);
        auto bytes = words.asBytes();
        //char* fff = (char*) bytes.begin();
        std::string data(bytes.begin(), bytes.end());
        //qDebug() << "BYTES SUZE" << bytes.size() << QString::fromStdString(data);
        qDebug()<<typeid(bytes).name();
        model_sock->send((char*)bytes.begin(), (size_t) bytes.size());

}
