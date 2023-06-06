import React, { useState, useEffect, useCallback } from "react";
import './view.css';

import OpenSeadragon from 'openseadragon';
import Spinner from 'react-bootstrap/Spinner';
import Dropdown from 'react-bootstrap/Dropdown';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import { useParams } from "react-router-dom";
import { SERVER_URL, LYMPHOMA_CLASSES } from '../constants';

function ViewPage() {
    let { id } = useParams();
    const [viewData, setViewData] = useState();
    const [errorMessage, setErrorMessage] = useState();
    const [loading, setLoading] = useState(true);
    const [probsListItems, setProbsListItems] = useState();
    const [chosenClassId, setChosenClassId] = useState(0);
    // const [viewer, setViewer] = useState();
    // const [viewerMap, setViewerMap] = useState();

    // let dzi_data = {}

    // function open_slide(url, mpp, viewerObj) {
    //     var tile_source;
    //     if (dzi_data[url]) {
    //         // DZI XML provided as template argument (deepzoom_tile.py)
    //         tile_source = new OpenSeadragon.DziTileSource(
    //             OpenSeadragon.DziTileSource.prototype.configure(
    //                 OpenSeadragon.parseXml(dzi_data[url]), url));
    //     } else {
    //         // DZI XML fetched from server (deepzoom_server.py)
    //         tile_source = url;
    //     }
    //     viewerObj.open(tile_source);
    //     // viewerObj.scalebar({
    //     //     pixelsPerMeter: mpp ? (1e6 / mpp) : 0,
    //     // });
    // }

    // const initViewers = async () => {
    //     if (viewData) {
    //         viewer && viewer.destroy();
    //         viewerMap && viewerMap.destroy();
    //         await setViewer(new OpenSeadragon({
    //             id: "view",
    //             prefixUrl: `${SERVER_URL}`,
    //             timeout: 120000,
    //             animationTime: 0.5,
    //             blendTime: 0.1,
    //             constrainDuringPan: true,
    //             maxZoomPixelRatio: 2,
    //             minZoomImageRatio: 1,
    //             visibilityRatio: 1,
    //             zoomPerScroll: 2,
    //         }));
    //         // viewer.scalebar({
    //         //     xOffset: 10,
    //         //     yOffset: 10,
    //         //     barThickness: 3,
    //         //     color: '#555555',
    //         //     fontColor: '#333333',
    //         //     backgroundColor: 'rgba(255, 255, 255, 0.5)',
    //         // });

    //         await setViewerMap(new OpenSeadragon({
    //             id: "viewmap",
    //             prefixUrl: `${SERVER_URL}`,
    //             timeout: 120000,
    //             animationTime: 0.5,
    //             blendTime: 0.1,
    //             constrainDuringPan: true,
    //             maxZoomPixelRatio: 2,
    //             minZoomImageRatio: 1,
    //             visibilityRatio: 1,
    //             zoomPerScroll: 2,
    //         }));

    //         open_slide(viewData.slide_url, viewData.slide_mpp, viewer);
    //         open_slide(viewData.map_urls[chosenClassId], viewData.slide_mpp, viewerMap);

    //         var viewerLeading = false;
    //         var viewerMapLeading = false;

    //         var viewerHandler = function () {
    //             if (viewerMapLeading) {
    //                 return;
    //             }

    //             viewerLeading = true;
    //             viewerMap.viewport.zoomTo(viewer.viewport.getZoom());
    //             viewerMap.viewport.panTo(viewer.viewport.getCenter());
    //             viewerLeading = false;
    //         };

    //         var viewerMapHandler = function () {
    //             if (viewerLeading) {
    //                 return;
    //             }

    //             viewerMapLeading = true;
    //             viewer.viewport.zoomTo(viewerMap.viewport.getZoom());
    //             viewer.viewport.panTo(viewerMap.viewport.getCenter());
    //             viewerMapLeading = false;
    //         };

    //         viewer.addHandler('zoom', viewerHandler);
    //         viewerMap.addHandler('zoom', viewerMapHandler);
    //         viewer.addHandler('pan', viewerHandler);
    //         viewerMap.addHandler('pan', viewerMapHandler);
    //     }
    // }

    function open_slide(url, mpp, viewerObj, dzi_data) {
        var tile_source;
        if (dzi_data[url]) {
            // DZI XML provided as template argument (deepzoom_tile.py)
            tile_source = new OpenSeadragon.DziTileSource(
                OpenSeadragon.DziTileSource.prototype.configure(
                    OpenSeadragon.parseXml(dzi_data[url]), url));
        } else {
            // DZI XML fetched from server (deepzoom_server.py)
            tile_source = url;
        }
        viewerObj.open(tile_source);
        viewerObj.scalebar({
            pixelsPerMeter: mpp ? (1e6 / mpp) : 0,
        });
    }

    const getData = useCallback(async () => {
        setLoading(true);
        setErrorMessage(null);
        setViewData(null);
        try {
            let response = await fetch(SERVER_URL + "/" + id);
            if (!response.ok) {
                throw Error(response.status);
            }

            let data = await response.json()

            setViewData(data);
            setLoading(false);

            try {
                let probsList = []
                for (let i = 0; i < data.class_probs.length; i++) {
                    if (data.class_probs[i] > 0.4)
                        probsList.push([data.class_probs[i], LYMPHOMA_CLASSES[i], i])
                }
                function compare(a, b) {
                    if (a[0] < b[0]) {
                        return 1;
                    }
                    if (a[0] > b[0]) {
                        return -1;
                    }
                    return 0;
                }

                probsList.sort(compare);

                setProbsListItems(probsList);

                let chosenClass = 0;

                if (probsList.length > 0) {
                    chosenClass = probsList[0][2]
                    setChosenClassId(chosenClass);
                }

                // let dzi_data = {};
                let viewer = await new OpenSeadragon({
                    id: "view",
                    prefixUrl: SERVER_URL,
                    timeout: 120000,
                    animationTime: 0.5,
                    blendTime: 0.1,
                    constrainDuringPan: true,
                    maxZoomPixelRatio: 2,
                    minZoomImageRatio: 1,
                    visibilityRatio: 1,
                    zoomPerScroll: 2,
                    tileSources: [`${SERVER_URL}/deepzoom/${id}.dzi`]
                });

                let viewerMap = await new OpenSeadragon({
                    id: "viewmap",
                    prefixUrl: SERVER_URL,
                    timeout: 120000,
                    animationTime: 0.5,
                    blendTime: 0.1,
                    constrainDuringPan: true,
                    maxZoomPixelRatio: 2,
                    minZoomImageRatio: 1,
                    visibilityRatio: 1,
                    zoomPerScroll: 2,
                    tileSources: [`${SERVER_URL}/deepzoom/${id}-${chosenClass}.dzi`]
                });

                // open_slide(data.slide_url, data.slide_mpp, viewer, dzi_data);
                // open_slide(data.map_urls[chosenClass], data.slide_mpp, viewerMap, dzi_data);

                let viewerLeading = false;
                let viewerMapLeading = false;

                let viewerHandler = function () {
                    if (viewerMapLeading) {
                        return;
                    }

                    viewerLeading = true;
                    viewerMap.viewport.zoomTo(viewer.viewport.getZoom());
                    viewerMap.viewport.panTo(viewer.viewport.getCenter());
                    viewerLeading = false;
                };

                let viewerMapHandler = function () {
                    if (viewerLeading) {
                        return;
                    }

                    viewerMapLeading = true;
                    viewer.viewport.zoomTo(viewerMap.viewport.getZoom());
                    viewer.viewport.panTo(viewerMap.viewport.getCenter());
                    viewerMapLeading = false;
                };

                viewer.addHandler('zoom', viewerHandler);
                viewerMap.addHandler('zoom', viewerMapHandler);
                viewer.addHandler('pan', viewerHandler);
                viewerMap.addHandler('pan', viewerMapHandler);
            }
            catch (e) {
            }
        }
        catch (e) {
            if (e.message === 404)
                setErrorMessage("Не знайдено зображення з цим ID.");
            else if (e.message === 400)
                setErrorMessage("Не знайдено результатів для цього зображення.");
            else
                setErrorMessage("Невідома помилка.");
            setLoading(false);
        }

    }, [id]);

    useEffect(() => {
        getData();
    }, [getData]);

    useEffect(() => {

    }, [])

    const chooseViewMap = (classId) => {
        setChosenClassId(classId);
    }


    return (
        <>
            {loading && <Spinner animation="border" role="status" />}
            {!loading && errorMessage && <p>Помилка при завантаженні: {errorMessage}</p>}
            {!loading && (viewData != null) && <>
                <h1>Аналіз зразка {id}</h1>
                {probsListItems && (probsListItems.length === 0 ? <>
                    <p>Система не розпізнає тут конкретний вид лімфоми...</p>
                </> : <>
                    <p>Найбільш імовірні діагнози лімфоми:</p>
                    <ul>
                        {probsListItems.map((pair) => {
                            return (<li key={pair[1]}>{pair[1]} - {(pair[0] * 100).toFixed(3)}% впевненості</li>);
                        })}
                    </ul>
                </>)}
                <div className="viewNames">
                    <div id="viewName"><p>Початкове повнослайдове зображення:</p></div>
                    <div id="viewmapname">
                        <Row>
                            <Col md="auto">
                                <span>Карти ознак для діагнозу:</span>
                            </Col>
                            <Col>
                                <Dropdown onSelect={chooseViewMap}>
                                    <Dropdown.Toggle variant="secondary">
                                        {LYMPHOMA_CLASSES[chosenClassId]}
                                    </Dropdown.Toggle>

                                    <Dropdown.Menu variant="dark">
                                        {LYMPHOMA_CLASSES.map((classname, index) =>
                                            <Dropdown.Item key={classname} eventKey={index} active={index === chosenClassId}>
                                                {classname}
                                            </Dropdown.Item>)}
                                    </Dropdown.Menu>
                                </Dropdown>
                            </Col>
                        </Row>

                    </div>
                </div>
                <div className="viewContainer">
                    <div id="view" className="view"></div>
                    <div id="viewcross" className='viewcross'></div>
                    <div id="viewmap" className="view"></div>
                    <div id="viewmapcross" className='viewcross'></div>
                </div>
            </>}
        </>
    );
}

export default ViewPage;