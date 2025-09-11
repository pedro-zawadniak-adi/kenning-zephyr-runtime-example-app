/*
 * Copyright (c) 2025 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kenning_inference_lib/core/model.h>
#include <kenning_inference_lib/core/utils.h>

#include "model_data.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#if defined(CONFIG_ZPL)
#include <zpl.h>
#endif

LOG_MODULE_REGISTER(app, CONFIG_APP_LOG_LEVEL);

#define ITER_EVAL_INTERVAL_MS 2000

int main(void)
{
    status_t status = STATUS_OK;
    uint8_t *model_input = NULL;
    uint8_t *model_output = NULL;
    size_t model_input_size = 0;
    size_t model_output_size = 0;
    int64_t timer_sec = 0;
    int64_t timer_start = 0;
    int64_t timer_end = 0;
    int64_t timer_iterations = 0;
    int32_t iterations = 0;

#if defined(CONFIG_ZPL)
    zpl_init();
#endif

    do
    {
        LOG_INF("Initializing model");
        // initialize model
        status = model_init();
        if (STATUS_OK != status)
        {
            LOG_ERR("Model initialization error 0x%x (%s)", status, get_status_str(status));
            break;
        }

        LOG_INF("Loading model_spec");
        // load model structure
        status = model_load_struct((uint8_t *)&model_spec, sizeof(model_spec_t));
        BREAK_ON_ERROR_LOG(status, "Model struct load error 0x%x (%s)", status, get_status_str(status));

        LOG_INF("Loading model_data");
        // load model weights
        status = model_load_weights(model_data, model_data_len);
        BREAK_ON_ERROR_LOG(status, "Model weights load error 0x%x (%s)", status, get_status_str(status));

        // allocate buffer for input
        model_get_input_size(&model_input_size);
        model_input = k_aligned_alloc(32, model_input_size);

        // allocate buffer for output
        model_get_output_size(&model_output_size);
        model_output = k_aligned_alloc(32, model_output_size);

        LOG_INF("Model is ready\n");

        while (true)
        {
            for (size_t i = 0; i < model_input_size; ++i)
            {
                model_input[i] = rand() & 0xff;
            }

            status = model_load_input(model_input, model_input_size);
            BREAK_ON_ERROR_LOG(status, "Model input load error 0x%x (%s)", status, get_status_str(status));

            timer_start = k_uptime_get();

            status = model_run();
            BREAK_ON_ERROR_LOG(status, "Model run error 0x%x (%s)", status, get_status_str(status));

            timer_end = k_uptime_get();

            status = model_get_output(model_output_size, model_output, NULL);
            BREAK_ON_ERROR_LOG(status, "Model get output error 0x%x (%s)", status, get_status_str(status));

            timer_iterations += (timer_end - timer_start);
            ++iterations;

            if (timer_end - timer_sec >= ITER_EVAL_INTERVAL_MS)
            {
                LOG_INF("Number of iterations:         %d", iterations);
                LOG_INF("Total time:                   %lld ms", timer_end - timer_sec);
                LOG_INF("Total time (inference only):  %lld ms", timer_iterations);
                LOG_INF("Average inference time:       %f ms", ((double)timer_iterations) / iterations);
                LOG_INF("Iterations per second:        %f\n\n", iterations / (double)timer_iterations * 1000);
                iterations = 0;
                timer_iterations = 0;
                timer_sec = k_uptime_get();
            }
        }
    } while (0);

    if (IS_VALID_POINTER(model_output))
    {
        free(model_output);
    }
    if (IS_VALID_POINTER(model_input))
    {
        free(model_input);
    }

    return 0;
}
