import React from "react";

import Form from 'react-bootstrap/Form';
import ProgressBar from 'react-bootstrap/ProgressBar';
import Spinner from 'react-bootstrap/Spinner';
import { Navigate } from 'react-router-dom';
import { SERVER_URL } from '../constants';

function UploadPage() {
    const [uploadProgress, setUploadProgress] = React.useState(0);
    const [showStatus, setShowStatus] = React.useState(false);
    const uploadRef = React.useRef();
    const [result, setResult] = React.useState();
    const [id, setId] = React.useState();

    const UploadFile = () => {
        const file = uploadRef.current.files[0];
        // setFile(URL.createObjectURL(file));
        var formData = new FormData();
        formData.append("file", file);
        var xhr = new XMLHttpRequest();
        setResult(null);
        setShowStatus(false);
        xhr.upload.addEventListener("progress", ProgressHandler, false);
        xhr.addEventListener("load", SuccessHandler, false);
        xhr.addEventListener("error", ErrorHandler, false);
        xhr.addEventListener("abort", AbortHandler, false);
        xhr.open("POST", SERVER_URL);
        xhr.send(formData);
    };

    const ProgressHandler = (e) => {
        var percent = (e.loaded / e.total) * 100;
        setUploadProgress(Math.round(percent));
    };

    const SuccessHandler = (e) => {
        var responseObj = JSON.parse(e.target.response)
        if (e.target.status_code === 200) {
            setResult(responseObj['message']);
            setId(responseObj['id']);
        }
        else {
            setResult(responseObj['message']);
        }
        setUploadProgress(0);
        setShowStatus(true);
    };

    const ErrorHandler = () => {
        setResult("Завантаження не вдалося з невідомих причин...");
        setShowStatus(true);
    };

    const AbortHandler = () => {
        setResult("Завантаження перервано...");
        setShowStatus(true);
    };

    if (id)
        return <Navigate to={'/' + id} />
    else return (
        <div className="mt-3">
            <Form.Group controlId="formFile" className="mb-3">
                <Form.Label>Завантажте повнослайдове зображення сюди:</Form.Label>
                <Form.Control type="file" ref={uploadRef} onChange={UploadFile} />
            </Form.Group>
            {(uploadProgress > 0) && (uploadProgress < 100) && (
                <>
                    <p>Відбувається завантаження файлу...</p>
                    <ProgressBar now={uploadProgress} label={`${uploadProgress}%`} />
                </>
            )}
            {(uploadProgress === 100) && !result && <p>Відбувається обробка зображення...</p>}
            {(uploadProgress === 100) && !result && <Spinner animation="border" role="status" />}
            {showStatus && result && <p>Результат обробки:</p>}
            {showStatus && result && <p>{result}</p>}
        </div>
    );
}

export default UploadPage;