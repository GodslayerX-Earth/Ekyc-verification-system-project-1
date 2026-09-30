#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {
std::filesystem::path resolve_model_path(const std::string& model_name) {
    const std::filesystem::path project_root = PROJECT_SOURCE_DIR;
    const std::vector<std::filesystem::path> candidates = {
        std::filesystem::current_path() / "models" / model_name,
        std::filesystem::current_path() / "../models" / model_name,
        project_root / "models" / model_name,
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return project_root / "models" / model_name;
}
}

int main() {
    const auto face_model_path = resolve_model_path("ultraface-320.onnx");
    const auto arc_model_path = resolve_model_path("arcface.onnx");

    if (!std::filesystem::exists(face_model_path) || !std::filesystem::exists(arc_model_path)) {
        std::cerr << "Missing model files. Checked:\n"
                  << "  - " << face_model_path << "\n"
                  << "  - " << arc_model_path << std::endl;
        return 1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "eKYC");
    Ort::SessionOptions opts; opts.SetIntraOpNumThreads(4);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Session face_session(env, face_model_path.string().c_str(), opts);
    Ort::Session arc_session(env, arc_model_path.string().c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    auto face_in_ptr = face_session.GetInputNameAllocated(0, alloc);
    const char* face_in[] = { face_in_ptr.get() };
    std::vector<Ort::AllocatedStringPtr> f_out_ptrs;
    std::vector<const char*> face_out;
    for(size_t i=0; i<face_session.GetOutputCount(); ++i) {
        f_out_ptrs.push_back(face_session.GetOutputNameAllocated(i, alloc));
        face_out.push_back(f_out_ptrs[i].get());
    }

    auto arc_in_ptr = arc_session.GetInputNameAllocated(0, alloc);
    const char* arc_in[] = { arc_in_ptr.get() };
    auto arc_out_ptr = arc_session.GetOutputNameAllocated(0, alloc);
    const char* arc_out[] = { arc_out_ptr.get() };

    cv::VideoCapture cap(0);
    cv::Mat frame, resized, float_img;
    std::vector<float> face_vals(1*3*240*320);
    std::vector<int64_t> face_shape = {1, 3, 240, 320};

    while (cap.read(frame)) {
        int w = frame.cols, h = frame.rows;
        cv::resize(frame, resized, cv::Size(320, 240));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(float_img, CV_32FC3, 1.0/128.0, -127.5/128.0);

        std::vector<cv::Mat> chw(3);
        for(int i=0; i<3; ++i) chw[i] = cv::Mat(240, 320, CV_32FC1, face_vals.data() + i*240*320);
        cv::split(float_img, chw);

        Ort::Value face_tensor = Ort::Value::CreateTensor<float>(mem_info, face_vals.data(), face_vals.size(), face_shape.data(), face_shape.size());
        auto f_out = face_session.Run(Ort::RunOptions{nullptr}, face_in, &face_tensor, 1, face_out.data(), face_out.size());
        
        float* scores = f_out[0].GetTensorMutableData<float>();
        float* boxes = f_out[1].GetTensorMutableData<float>();

        std::vector<cv::Rect> c_boxes; std::vector<float> c_confs;
        for (int i = 0; i < 4420; ++i) {
            if (scores[i*2+1] > 0.7f) {
                int x1 = boxes[i*4+0]*w, y1 = boxes[i*4+1]*h, x2 = boxes[i*4+2]*w, y2 = boxes[i*4+3]*h;
                c_boxes.push_back(cv::Rect(x1, y1, x2-x1, y2-y1));
                c_confs.push_back(scores[i*2+1]);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(c_boxes, c_confs, 0.7f, 0.3f, indices); // NMS ensures ONLY ONE BOX

        for (int idx : indices) {
            cv::Rect b = c_boxes[idx];
            b &= cv::Rect(0, 0, w, h);
            if (b.empty()) {
                continue;
            }

            cv::Mat crop = frame(b).clone(), arc_res, arc_f;
            cv::resize(crop, arc_res, cv::Size(112, 112));
            cv::cvtColor(arc_res, arc_res, cv::COLOR_BGR2RGB);
            arc_res.convertTo(arc_f, CV_32FC3, 1.0/127.5, -1.0);

            std::vector<int64_t> a_shape = {1, 112, 112, 3};
            Ort::Value a_tensor = Ort::Value::CreateTensor<float>(
                mem_info,
                arc_f.ptr<float>(),
                1 * 112 * 112 * 3,
                a_shape.data(),
                a_shape.size());
            auto a_out = arc_session.Run(Ort::RunOptions{nullptr}, arc_in, &a_tensor, 1, arc_out, 1);
            
            float* emb = a_out[0].GetTensorMutableData<float>();

            const double time = cv::getTickCount() / cv::getTickFrequency();
            const double pulse = 1.0 + 0.035 * std::sin(time * 5.0);
            const cv::Point center(b.x + b.width / 2, b.y + b.height / 2);
            const cv::Size axes(
                std::max(1, static_cast<int>(b.width * 0.53 * pulse)),
                std::max(1, static_cast<int>(b.height * 0.56 * pulse)));
            const cv::Scalar cyan(255, 255, 0);

            cv::ellipse(frame, center, axes, 0, 0, 360, cyan, 2, cv::LINE_AA);
            cv::ellipse(frame, center, axes, 0,
                        std::fmod(time * 140.0, 360.0),
                        std::fmod(time * 140.0 + 75.0, 360.0),
                        cv::Scalar(255, 255, 255), 3, cv::LINE_AA);

            std::vector<std::vector<cv::Point>> mesh_rows;
            constexpr int row_count = 9;
            constexpr int points_per_row = 13;
            for (int row = 0; row < row_count; ++row) {
                const double normalized_y = -0.82 + row * 1.64 / (row_count - 1);
                const double row_width = std::sqrt(std::max(0.0, 1.0 - normalized_y * normalized_y));
                std::vector<cv::Point> points;
                for (int column = 0; column < points_per_row; ++column) {
                    const double normalized_x = -0.84 + column * 1.68 / (points_per_row - 1);
                    if (std::abs(normalized_x) > row_width) {
                        continue;
                    }
                    points.emplace_back(
                        center.x + static_cast<int>(normalized_x * axes.width * 0.92),
                        center.y + static_cast<int>(normalized_y * axes.height * 0.92));
                }
                mesh_rows.push_back(std::move(points));
            }

            for (const auto& row : mesh_rows) {
                for (size_t point = 1; point < row.size(); ++point) {
                    cv::line(frame, row[point - 1], row[point], cyan, 1, cv::LINE_AA);
                }
                for (const cv::Point& landmark : row) {
                    cv::circle(frame, landmark, 2, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
                }
            }
            for (size_t row = 1; row < mesh_rows.size(); ++row) {
                const auto& previous = mesh_rows[row - 1];
                const auto& current = mesh_rows[row];
                for (size_t point = 0; point < current.size(); ++point) {
                    const size_t matching_point = std::min(point, previous.size() - 1);
                    cv::line(frame, previous[matching_point], current[point], cyan, 1, cv::LINE_AA);
                }
            }

            char buf[64];
            snprintf(buf, sizeof(buf), "512d: [%.2f, %.2f...]", emb[0], emb[1]);
            cv::putText(frame, buf, cv::Point(b.x, b.y + b.height + 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cyan, 2, cv::LINE_AA);
        }

        cv::imshow("eKYC - Mesh & Single Box", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}
