import vtk
import numpy as np
from sklearn.cluster import KMeans
from skimage import morphology
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import time

def long_running_task(total_steps):
    #print("Task started...")
    start_time = time.time()  
    
    for step in range(1, total_steps + 1):
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / step) * (total_steps - step)
        #print(f"Step {step}/{total_steps} completed. Estimated time left: {remaining_time:.2f} seconds")

    #print("Task completed!")

def process_dicom(directory):
    
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(directory)
    reader.Update()

    long_running_task(1)  

    imageData = reader.GetOutput()

    extent = imageData.GetExtent()
    spacing = imageData.GetSpacing()
    dimensions = (extent[1] - extent[0] + 1, extent[3] - extent[2] + 1, extent[5] - extent[4] + 1)
    voxelData = vtk_to_numpy(imageData.GetPointData().GetScalars())
    voxelData = voxelData.reshape(dimensions, order='F')

    voxelData = (voxelData - np.min(voxelData)) / (np.max(voxelData) - np.min(voxelData))

    voxelFlat = voxelData.ravel().reshape(-1, 1)

    kmeans = KMeans(n_clusters=3, random_state=0)  
    kmeans.fit(voxelFlat)
    clustered = kmeans.labels_.reshape(voxelData.shape)

    cluster_means = [voxelData[clustered == i].mean() for i in range(3)]
    teeth_cluster = np.argmax(cluster_means)  

    teeth_mask = (clustered == teeth_cluster).astype(np.uint8)

    teeth_mask = morphology.binary_closing(teeth_mask, morphology.ball(3))
    teeth_mask = morphology.binary_opening(teeth_mask, morphology.ball(2))

    segmentedArray = teeth_mask.astype(np.uint8)
    segmentedVTKData = numpy_to_vtk(num_array=segmentedArray.ravel(order='F'), deep=True)
    segmentedImageData = vtk.vtkImageData()
    segmentedImageData.SetDimensions(dimensions)
    segmentedImageData.SetSpacing(spacing)
    segmentedImageData.GetPointData().SetScalars(segmentedVTKData)

    long_running_task(3)  

    resampler = vtk.vtkImageReslice()
    resampler.SetInputData(segmentedImageData)
    resampler.SetOutputSpacing(0.180311183, 0.180311183, 0.18031118)
    resampler.SetInterpolationModeToLinear()
    resampler.Update()

    resampledData = resampler.GetOutput()

    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetInputData(resampledData)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()

    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(0, 0.0)
    opacityTransferFunction.AddPoint(1, 1.0)  

    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(0, 0.0, 0.0, 0.0)  
    colorTransferFunction.AddRGBPoint(1, 1.0, 1.0, 1.0)  

    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.SetColor(colorTransferFunction)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0, 0, 0)  

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(800, 800)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    xPlaneWidget = vtk.vtkImplicitPlaneWidget2()
    xPlaneRep = vtk.vtkImplicitPlaneRepresentation()
    xPlaneRep.SetPlaceFactor(1.25)
    xPlaneRep.PlaceWidget(volume.GetBounds())
    xPlaneRep.SetNormal(1, 0, 0)  
    xPlaneWidget.SetRepresentation(xPlaneRep)
    xPlaneWidget.SetInteractor(renderWindowInteractor)

    yPlaneWidget = vtk.vtkImplicitPlaneWidget2()
    yPlaneRep = vtk.vtkImplicitPlaneRepresentation()
    yPlaneRep.SetPlaceFactor(1.25)
    yPlaneRep.PlaceWidget(volume.GetBounds())
    yPlaneRep.SetNormal(0, 1, 0)  
    yPlaneWidget.SetRepresentation(yPlaneRep)
    yPlaneWidget.SetInteractor(renderWindowInteractor)

    xPlane = vtk.vtkPlane()
    yPlane = vtk.vtkPlane()

    def update_clipping_planes():
        clippingPlanes = vtk.vtkPlaneCollection()
        clippingPlanes.AddItem(xPlane)
        clippingPlanes.AddItem(yPlane)
        volumeMapper.SetClippingPlanes(clippingPlanes)

    def x_plane_callback(widget, event):
        xPlaneRep.GetPlane(xPlane)
        update_clipping_planes()

    def y_plane_callback(widget, event):
        yPlaneRep.GetPlane(yPlane)
        update_clipping_planes()

    xPlaneWidget.AddObserver("InteractionEvent", x_plane_callback)
    yPlaneWidget.AddObserver("InteractionEvent", y_plane_callback)

    xPlaneWidget.On()
    yPlaneWidget.On()

    camera = renderer.GetActiveCamera()
    camera.SetViewUp(0, 0, -1)
    camera.SetPosition(-500, -500, 500)
    camera.SetFocalPoint(0, 0, 0)
    renderer.ResetCamera()

    #print("Rendering completed.")
    
    renderWindow.Render()
    renderWindowInteractor.Start()

if __name__ == "__main__":
    process_dicom('Data/raw')