#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <deque>


namespace py = pybind11;

py::array_t<double> event_queue_tensor(py::array_t<float> events, int queue_length, int B, int H, int W, py::array_t<int> start_times, int event_window_abin) 
{
    // find number of events
    auto events_buf = events.request();
    float *events_ptr = (float *) events_buf.ptr;
    auto start_times_buf = start_times.request();
    int *start_times_ptr = (int *) start_times_buf.ptr;
    int n_events = events_buf.shape[0];

    /*  allocate the buffer */
    py::array_t<float> result = py::array_t<float>({2, queue_length, 2, B, H, W});
    py::array_t<float> result_now = py::array_t<float>({2, B, H, W});
    py::array_t<float> ecd_now = py::array_t<float>({2, B, H, W});
    //py::array_t<int> queue_size = py::array_t<int>({H, W});
    
    auto result_buf = result.request();
    float *result_ptr = (float *) result_buf.ptr;
    auto result_now_buf = result_now.request();
    float *result_now_ptr = (float *) result_now_buf.ptr;
    auto ecd_now_buf = ecd_now.request();
    float *ecd_now_ptr = (float *) ecd_now_buf.ptr;

    std::vector<std::deque<float>> result_deque(2*B*H*W);
    std::vector<std::deque<float>> ecd_deque(2*B*H*W);

    //auto queue_size_buf = queue_size.request();
    //double *queue_size_ptr = (double *) queue_size_buf.ptr;
    for (int i=0; i<2*B*H*W; i++)
    {
        result_now_ptr[i] = 0;
        ecd_now_ptr[i] = -1;
    }

    for (int idx = 0; idx < n_events; idx++)
    {
        int b = events_ptr[6 * idx + 0];
        int w = events_ptr[6 * idx + 1];
        int h = events_ptr[6 * idx + 2];
        float t = events_ptr[6 * idx + 3];
        int p = events_ptr[6 * idx + 4];
        int z = events_ptr[6 * idx + 5];
        bool add_forward = false;

        int queue_size = ecd_deque[B*H*W*p+H*W*b+W*h+w].size();

        if ((queue_size > 0)&(z > ecd_now_ptr[B*H*W*p+H*W*b+W*h+w]))
        {
            result_deque[B*H*W*p+H*W*b+W*h+w].push_front(result_now_ptr[B*H*W*p+H*W*b+W*h+w]);
            ecd_deque[B*H*W*p+H*W*b+W*h+w].push_front(ecd_now_ptr[B*H*W*p+H*W*b+W*h+w]);
            result_now_ptr[B*H*W*p+H*W*b+W*h+w] = 0;
            ecd_now_ptr[B*H*W*p+H*W*b+W*h+w] = z;
        }

        if (queue_size > 0)
        {
            if (z == ecd_deque[B*H*W*p+H*W*b+W*h+w].front() + 1) add_forward = true; 
            result_now_ptr[B*H*W*p+H*W*b+W*h+w] += 1 - (start_times_ptr[b] + event_window_abin * (z + 1) - t)/event_window_abin;
            if (add_forward) result_deque[B*H*W*p+H*W*b+W*h+w].front() += 1 - (t - (start_times_ptr[b] + event_window_abin * z))/event_window_abin;
        }
        else result_now_ptr[B*H*W*p+H*W*b+W*h+w] += 1 - (start_times_ptr[b] + event_window_abin * (z + 1) - t)/event_window_abin;
            
        if (result_deque[B*H*W*p+H*W*b+W*h+w].size() > queue_length)
        {
            result_deque[B*H*W*p+H*W*b+W*h+w].pop_back();
            ecd_deque[B*H*W*p+H*W*b+W*h+w].pop_back();
        }
            
    }

    for (int i=0; i<2*B*H*W; i++)
    {
        if (result_now_ptr[i] > 0)
        {
            result_deque[i].push_front(result_now_ptr[i]);
            ecd_deque[i].push_front(ecd_now_ptr[i]);
            if (result_deque[i].size() > queue_length)
            {
                result_deque[i].pop_back();
                ecd_deque[i].pop_back();
            }
        }
    }

    for (int i=0; i<2*B*H*W; i++)
    {
        auto& r = result_deque[i];
        auto& e = ecd_deque[i];
        for (int k=queue_length - 1; k>=0; k--)
        {
            if (!e.empty())
            {
                float &p1 = r.front();
                //std::cout << " p.second " << p.second << std::endl;
                result_ptr[i+2*B*H*W*k] = p1;
                float &p2 = e.front();
                result_ptr[i+2*B*H*W*k+2*B*H*W*queue_length] = p2;
                r.pop_front();
                e.pop_front();
            }
            else
            {
                result_ptr[i+2*B*H*W*k] = 0;
                result_ptr[i+2*B*H*W*k+2*B*H*W*queue_length] = -1;
            }
        }
    }
    
    return result;
}

PYBIND11_MODULE(event_representations, m) {
        m.doc() = "Generate event representations"; // optional module docstring
        m.def("event_queue_tensor", &event_queue_tensor, "Generate event queue tensor");
}