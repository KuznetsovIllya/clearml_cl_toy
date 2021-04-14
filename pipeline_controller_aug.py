from clearml import Task
from clearml.automation.controller import PipelineController


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='rabbit_fox', task_name='test pipeline',
                 task_type=Task.TaskTypes.controller, reuse_last_task_id=False)

pipe = PipelineController(default_execution_queue='default', add_pipeline_tags=False)
pipe.add_step(name='image_augmentation_copy', base_task_project='rabbit_fox', base_task_name='image_augmentation')
pipe.add_step(name='train_1st_nn_copy', parents=['image_augmentation_copy', ], base_task_project='rabbit_fox', base_task_name='train_1st_nn', parameter_override={'batch_size': 8})
pipe.add_step(name='train_2nd_nn_copy', parents=['train_1st_nn_copy', ],
              base_task_project='rabbit_fox', base_task_name='train_2nd_nn',
              parameter_override={'batch_size': 4})

# Starting the pipeline (in the background)
pipe.start()
# Wait until pipeline terminates
pipe.wait()
# cleanup everything
pipe.stop()

print('done')